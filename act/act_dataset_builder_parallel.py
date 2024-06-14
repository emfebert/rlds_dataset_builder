from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from example_dataset.conversion_utils import MultiThreadedDatasetBuilder
import h5py
from skimage.transform import resize
from scipy.spatial.transform import Rotation

IMSIZE = 256


def euler_to_r6(euler, degrees=False):
    rot_mat = Rotation.from_euler("xyz", euler, degrees=degrees).as_matrix()
    a1, a2 = rot_mat[0], rot_mat[1]
    return np.concatenate((a1, a2)).astype(np.float32)

def get_qpos(root):
    # goes together with the function below! Do not change separately!
    xyz = root['observations']['robot_poses'][:, :3]  # get the xyz
    joint_angles = root['observations']['robot_poses'][:, 3:]  # get the joint angles and convert to r6
    r6s = np.array([euler_to_r6(degrees, degrees=True) for degrees in joint_angles])
    return np.concatenate([xyz, r6s], axis=1)

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Generator of examples for each split."""

    # the line below needs to be *inside* generate_examples so that each worker creates it's own model
    # creating one shared model outside this function would cause a deadlock
    _embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _parse_example(episode_path):
        
        with h5py.File(episode_path, 'r') as root:

            if 'language_instruction' in root:
                lanugage_instruction = root['language_instruction'][()]
            else:
                lanugage_instruction = "blank"
            
            actions = np.array(root['/action'])
            logger.info(f"actions shape {actions.shape}")

            logger.info(f"image keys {root['/observations/images'].keys()}")

            wrist_images = np.array(root['/observations/images/color_image'])

            logger.info(f"wrist_images shape {wrist_images.shape}")


            # get observation at start_ts only
            qpos = get_qpos(root)
            logger.info(f"qpos shape {qpos.shape}")

            T = root['/observations/robot_poses'].shape[0]

        logger.info(f"T {T}")
        logger.info(f"qpos shape {qpos.shape}")
        logger.info(f"actions shape {actions.shape}")
        logger.info(f"wrist_images shape {wrist_images.shape}")

        # copy the last action at the end of the action sequence
        # the assumption is that s_t+1 = step(s_t, a_t), the first state is s_0=reset(), so the last action is not used
        actions = np.concatenate([actions, actions[-1][None, :]], axis=0)

        # resample image to 256x256
        wrist_images = resize(
                wrist_images, (IMSIZE, IMSIZE), anti_aliasing=True
            )

        
        # assemble episode --> here we're assuming demos so we set reward to 1 at the end
        episode = []
        for i in range(T):
            # compute Kona language embedding
            language_embedding = _embed([lanugage_instruction])[0].numpy()


            episode.append({
                'observation': {
                    'image_wrist': wrist_images[i],
                    'state': qpos[i],
                },
                'action': actions[i],
                'discount': 1.0,
                'reward': float(i == (T - 1)),
                'is_first': i == 0,
                'is_last': i == (T - 1),
                'is_terminal': i == (T - 1),
                'language_instruction': lanugage_instruction,
                'language_embedding': language_embedding,
            })

        # create output data sample
        sample = {
            'steps': episode,
            'episode_metadata': {
                'file_path': episode_path
            }
        }

        # if you want to skip an example for whatever reason, simply return None
        return episode_path, sample

    # for smallish datasets, use single-thread parsing
    for sample in paths:
        yield _parse_example(sample)


class ExampleDataset(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }
    N_WORKERS = 10             # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 100  # number of paths converted & stored in memory before writing to disk
                               # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
                               # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = _generate_examples      # handle to parse function from file paths to RLDS episodes

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image_wrist': tfds.features.Image(
                            shape=(IMSIZE, IMSIZE, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(9,),
                            dtype=np.float32,
                            doc='Robot state, consists of [7x robot joint angles, '
                                '2x gripper position, 1x door opening angle].',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(10,),
                        dtype=np.float32,
                        doc='Robot action, consists of [7x joint velocities, '
                            '2x gripper velocities, 1x terminate episode].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_paths(self):
        """Define filepaths for data splits."""
        print(self.info)
        paths = glob.glob('/home/febert/data/closed_loop_demos/insert_ibuprofen/*/episode*/*.hdf5')
        logger.log(f"num paths {len(paths)}")
        return {
            'train': paths,
        }

