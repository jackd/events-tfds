"""hmdb_dvs dataset."""
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from scipy.io import loadmat

GRID_SHAPE = (128, 128)


def load_event_data(path: str):
    """Load (time, coords, polarity) from .mat path."""
    data = loadmat(path)
    time, x, y, polarity = (
        np.squeeze(data[k], axis=-1) for k in ("ts", "x", "y", "pol")
    )
    coords = np.stack((x, y), axis=-1)
    return time, coords, polarity


CLASSES = (
    "ApplyEyeMakeup",
    "ApplyLipstick",
    "Archery",
    "BabyCrawling",
    "BalanceBeam",
    "BandMarching",
    "BaseballPitch",
    "Basketball",
    "BasketballDunk",
    "BenchPress",
    "Biking",
    "Billiards",
    "BlowDryHair",
    "BlowingCandles",
    "BodyWeightSquats",
    "Bowling",
    "BoxingPunchingBag",
    "BoxingSpeedBag",
    "BreastStroke",
    "BrushingTeeth",
    "CleanAndJerk",
    "CliffDiving",
    "CricketBowling",
    "CricketShot",
    "CuttingInKitchen",
    "Diving",
    "Drumming",
    "Fencing",
    "FieldHockeyPenalty",
    "FloorGymnastics",
    "FrisbeeCatch",
    "FrontCrawl",
    "GolfSwing",
    "Haircut",
    "HammerThrow",
    "Hammering",
    "HandstandPushups",
    "HandstandWalking",
    "HeadMassage",
    "HighJump",
    "HorseRace",
    "HorseRiding",
    "HulaHoop",
    "IceDancing",
    "JavelinThrow",
    "JugglingBalls",
    "JumpRope",
    "JumpingJack",
    "Kayaking",
    "Knitting",
    "LongJump",
    "Lunges",
    "MilitaryParade",
    "Mixing",
    "MoppingFloor",
    "Nunchucks",
    "ParallelBars",
    "PizzaTossing",
    "PlayingCello",
    "PlayingDaf",
    "PlayingDhol",
    "PlayingFlute",
    "PlayingGuitar",
    "PlayingPiano",
    "PlayingSitar",
    "PlayingTabla",
    "PlayingViolin",
    "PoleVault",
    "PommelHorse",
    "PullUps",
    "Punch",
    "PushUps",
    "Rafting",
    "RockClimbingIndoor",
    "RopeClimbing",
    "Rowing",
    "SalsaSpin",
    "ShavingBeard",
    "Shotput",
    "SkateBoarding",
    "Skiing",
    "Skijet",
    "SkyDiving",
    "SoccerJuggling",
    "SoccerPenalty",
    "StillRings",
    "SumoWrestling",
    "Surfing",
    "Swing",
    "TableTennisShot",
    "TaiChi",
    "TennisSwing",
    "ThrowDiscus",
    "TrampolineJumping",
    "Typing",
    "UnevenBars",
    "VolleyballSpiking",
    "WalkingWithDog",
    "WallPushups",
    "WritingOnBoard",
    "YoYo",
)
NUM_CLASSES = len(CLASSES)
assert NUM_CLASSES == 101


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for ucf101_dvs dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    MANUAL_DOWNLOAD_INSTRUCTIONS = (
        "Download UCF101-DVS files from "
        "https://www.dropbox.com/sh/ie75dn246cacf6n/AACoU-_zkGOAwj51lSCM0JhGa?dl=0"
    )

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "events": tfds.features.FeaturesDict(  # tfds.features.Sequence(
                        {
                            "time": tfds.features.Tensor(shape=(None,), dtype=np.int64),
                            "coords": tfds.features.Tensor(
                                shape=(None, 2),
                                dtype=np.int64,
                            ),
                            "polarity": tfds.features.Tensor(
                                shape=(None,), dtype=np.bool_
                            ),
                        }
                    ),
                    "label": tfds.features.ClassLabel(names=CLASSES),
                }
            ),
            supervised_keys=("events", "label"),
            homepage="https://github.com/PIX2NVS/NVS2Graph",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # root_archive = dl_manager.download(
        #     "https://www.dropbox.com/sh/ie75dn246cacf6n/AACoU-_zkGOAwj51lSCM0JhGa?dl=1"
        # )
        # root_archive = dl_manager.extract(root_archive)
        # root_dirs = []
        # for subdir in tf.io.gfile.listdir(root_archive):
        #     if "UCF101-DVS" in subdir:
        #         root_dirs.append(root_archive / subdir)
        # root_dirs = dl_manager.extract(root_dirs)
        manual_dir = dl_manager.manual_dir
        archives = [
            manual_dir / fn
            for fn in tf.io.gfile.listdir(manual_dir)
            if "UCF101-DVS" in fn
        ]
        assert len(archives) == 5, archives
        archive_iters = [dl_manager.iter_archive(archive) for archive in archives]
        return {
            "train": self._generate_examples(archive_iters),
        }

    # def _generate_examples(self, root_dirs: tp.Iterable[Path]):
    #     """Yields examples."""
    #     for root_dir in root_dirs:
    #         for label in tf.io.gfile.listdir(root_dir):
    #             for filename in tf.io.gfile.listdir(root_dir / label):
    #                 path = root_dir / label / filename
    #                 time, coords, polarity = load_event_data(path)
    #                 if time.shape[0] == 0:
    #                     continue
    #                 features = {
    #                     "events": {
    #                         "time": time.astype(np.int64),
    #                         "coords": coords.astype(np.int64),
    #                         "polarity": polarity.astype(bool),
    #                     },
    #                     "label": label,
    #                 }
    #                 yield filename, features
    def _generate_examples(self, archive_iters):
        """Yields examples."""
        for archive_iter in archive_iters:
            for path, fileobj in archive_iter:
                label, filename = path.split("/")
                time, coords, polarity = load_event_data(fileobj)
                if time.shape[0] == 0:
                    continue
                features = {
                    "events": {
                        "time": time.astype(np.int64),
                        "coords": coords.astype(np.int64),
                        "polarity": polarity.astype(bool),
                    },
                    "label": label,
                }
                yield filename, features
