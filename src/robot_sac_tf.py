import tensorflow as tf
import tempfile
import os

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.environments import suite_pybullet
from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils
from tf_agents.utils import nest_utils
from typing import Sequence


class RobotSacTf:
    def __init__(
        self,
        env: PyEnvironment,
        fc_layer_params: Sequence[int],
    ):
        import os

        # Keep using keras-2 (tf-keras) rather than keras-3 (keras).
        os.environ["TF_USE_LEGACY_KERAS"] = "1"
        self._env = env
        use_gpu = True
        self._strategy = strategy_utils.get_strategy(tpu=False, use_gpu=use_gpu)
        observation_spec, action_spec, time_step_spec = spec_utils.get_tensor_specs(
            self._env
        )

        with self._strategy.scope():
            self._critic_net = critic_network.CriticNetwork(
                (observation_spec, action_spec),
                observation_fc_layer_params=None,
                action_fc_layer_params=None,
                joint_fc_layer_params=fc_layer_params,
                kernel_initializer="glorot_uniform",
                last_kernel_initializer="glorot_uniform",
            )
            self._actor_net = actor_distribution_network.ActorDistributionNetwork(
                observation_spec,
                action_spec,
                fc_layer_params=fc_layer_params,
                continuous_projection_net=(
                    tanh_normal_projection_network.TanhNormalProjectionNetwork
                ),
            )

    def learn(
        self,
        num_iterations=100000,
        learning_rate=3e-4,
        gamma=0.99,
        target_update_tau=0.005,
        target_update_period=1,
        replay_buffer_capacity=10_000_000,
        batch_size=256,
        initial_collect_steps=1000,
        tempdir=tempfile.mkdtemp("tfagents"),
        num_eval_episodes=20,
        eval_interval=5000,
        log_interval=1000,
        policy_save_interval=5000,
    ):
        observation_spec, action_spec, time_step_spec = spec_utils.get_tensor_specs(
            self._env
        )
        with self._strategy.scope():
            train_step = train_utils.create_train_step()
            tf_agent = sac_agent.SacAgent(
                time_step_spec,
                action_spec,
                actor_network=self._actor_net,
                critic_network=self._critic_net,
                actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=learning_rate
                ),
                critic_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=learning_rate
                ),
                alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=learning_rate
                ),
                target_update_tau=target_update_tau,
                target_update_period=target_update_period,
                td_errors_loss_fn=tf.math.squared_difference,
                gamma=gamma,
                reward_scale_factor=1.0,
                gradient_clipping=None,
                debug_summaries=False,
                summarize_grads_and_vars=False,
                train_step_counter=train_step,
            )
            tf_agent.initialize()

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            tf_agent.collect_data_spec,
            batch_size=1,
            max_length=replay_buffer_capacity,
        )
        replay_observe = lambda traj: replay_buffer.add_batch(nest_utils.batch_nested_array(traj))
        dataset = replay_buffer.as_dataset(
            sample_batch_size=batch_size, num_steps=2
        ).prefetch(50)
        experience_dataset_fn = lambda: dataset
        tf_eval_policy = tf_agent.policy
        self._eval_policy = py_tf_eager_policy.PyTFEagerPolicy(
            tf_eval_policy, use_tf_function=True
        )
        tf_collect_policy = tf_agent.collect_policy
        collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
            tf_collect_policy, use_tf_function=True
        )
        random_policy = random_py_policy.RandomPyPolicy(time_step_spec, action_spec)

        initial_collect_actor = actor.Actor(
            self._env,
            random_policy,
            train_step,
            steps_per_run=initial_collect_steps,
            observers=[replay_observe],
        )
        print("about to run initial_collect_actor")
        initial_collect_actor.run()
        print("ran initial_collect_actor")
        env_step_metric = py_metrics.EnvironmentSteps()
        collect_actor = actor.Actor(
            self._env,
            collect_policy,
            train_step,
            steps_per_run=1,
            metrics=actor.collect_metrics(10),
            summary_dir=os.path.join(tempdir, learner.TRAIN_DIR),
            observers=[replay_observe, env_step_metric],
        )
        eval_actor = actor.Actor(
            self._env,
            self._eval_policy,
            train_step,
            episodes_per_run=num_eval_episodes,
            metrics=actor.eval_metrics(num_eval_episodes),
            summary_dir=os.path.join(tempdir, "eval"),
        )
        saved_model_dir = os.path.join(tempdir, learner.POLICY_SAVED_MODEL_DIR)

        print("made it up to saved_model_dir")

        # Triggers to save the agent's policy checkpoints.
        learning_triggers = [
            triggers.PolicySavedModelTrigger(
                saved_model_dir, tf_agent, train_step, interval=policy_save_interval
            ),
            triggers.StepPerSecondLogTrigger(train_step, interval=1000),
        ]

        agent_learner = learner.Learner(
            tempdir,
            train_step,
            tf_agent,
            experience_dataset_fn,
            triggers=learning_triggers,
            strategy=self._strategy,
        )

        def get_eval_metrics():
            eval_actor.run()
            results = {}
            for metric in eval_actor.metrics:
                results[metric.name] = metric.result()
            return results

        metrics = get_eval_metrics()

        def log_eval_metrics(step, metrics):
            eval_results = (", ").join(
                "{} = {:.6f}".format(name, result) for name, result in metrics.items()
            )
            print("step = {0}: {1}".format(step, eval_results))

        log_eval_metrics(0, metrics)
        tf_agent.train_step_counter.assign(0)

        # Evaluate the agent's policy once before training.
        avg_return = get_eval_metrics()["AverageReturn"]
        returns = [avg_return]

        tf.autograph.set_verbosity(10)

        for _ in range(num_iterations):
            # Training.
            collect_actor.run()
            loss_info = agent_learner.run(iterations=1)

            # Evaluating.
            step = agent_learner.train_step_numpy

            if eval_interval and step % eval_interval == 0:
                metrics = get_eval_metrics()
                log_eval_metrics(step, metrics)
                returns.append(metrics["AverageReturn"])

            if log_interval and step % log_interval == 0:
                print("step = {0}: loss = {1}".format(step, loss_info.loss.numpy()))

