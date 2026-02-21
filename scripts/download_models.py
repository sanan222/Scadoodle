import os
import shutil
from pathlib import Path
from abc import ABC, abstractmethod
import torch
from ultralytics import YOLO, FastSAM, SAM
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights


class ModelDownloader(ABC):
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.models_dir.mkdir(exist_ok=True)
    
    @abstractmethod
    def download(self) -> bool:
        pass
    
    def _copy_checkpoint(self, source: Path, destination: Path) -> bool:
        if source.exists():
            shutil.copy(source, destination)
            return True
        return False


class YOLODownloader(ModelDownloader):
    def download(self) -> bool:
        try:
            model_yolov10n = YOLO('yolov10n.pt')
            model_yolov9c = YOLO('yolov9c.pt')
            
            sources = [
                Path.home() / '.config/Ultralytics/yolov10n.pt',
                Path('yolov10n.pt')
            ]
            for src in sources:
                if self._copy_checkpoint(src, self.models_dir / 'yolov10n.pt'):
                    break
            
            sources = [
                Path.home() / '.config/Ultralytics/yolov9c.pt',
                Path('yolov9c.pt')
            ]
            for src in sources:
                if self._copy_checkpoint(src, self.models_dir / 'yolov9c.pt'):
                    break
            
            return True
        except Exception:
            return False


class SAMDownloader(ModelDownloader):
    def download(self) -> bool:
        try:
            model_fastsam = FastSAM('FastSAM-x.pt')
            model_mobilesam = SAM('mobile_sam.pt')
            
            fastsam_sources = [
                Path.home() / '.config/Ultralytics/FastSAM-x.pt',
                Path('FastSAM-x.pt')
            ]
            for src in fastsam_sources:
                if self._copy_checkpoint(src, self.models_dir / 'fastsam-x.pt'):
                    break
            
            mobilesam_sources = [
                Path.home() / '.config/Ultralytics/mobile_sam.pt',
                Path('mobile_sam.pt')
            ]
            for src in mobilesam_sources:
                if self._copy_checkpoint(src, self.models_dir / 'mobile_sam.pt'):
                    break
            
            return True
        except Exception:
            return False


class RAFTDownloader(ModelDownloader):
    def download(self) -> bool:
        try:
            weights = Raft_Small_Weights.DEFAULT
            model = raft_small(weights=weights)
            torch.save(model.state_dict(), self.models_dir / 'raft_small.pth')
            return True
        except Exception:
            return False


class MediaPipeDownloader(ModelDownloader):
    def download(self) -> bool:
        note_path = self.models_dir / 'MEDIAPIPE_README.txt'
        note_path.write_text(
            "MediaPipe models are automatically managed by the library.\n"
            "Models are cached internally by MediaPipe.\n"
            "No manual checkpoint management needed.\n"
            "\nAvailable tasks:\n"
            "- MediaPipe Hands: Hand landmark detection (21 landmarks)\n"
            "- MediaPipe Pose: Full body pose detection (33 landmarks)\n"
        )
        return True


class TrackingDownloader(ModelDownloader):
    def download(self) -> bool:
        note_path = self.models_dir / 'TRACKING_README.txt'
        note_path.write_text(
            "ByteTrack and BotSORT tracking algorithms are built into Ultralytics.\n"
            "No separate model checkpoints needed.\n"
            "\nUsage:\n"
            "from ultralytics import YOLO\n"
            "model = YOLO('yolov10n.pt')\n"
            "results = model.track(source='video.mp4', tracker='bytetrack.yaml')\n"
        )
        return True


class ModelManager:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.downloaders = [
            YOLODownloader(models_dir),
            SAMDownloader(models_dir),
            RAFTDownloader(models_dir),
            MediaPipeDownloader(models_dir),
            TrackingDownloader(models_dir)
        ]
    
    def download_all(self):
        for downloader in self.downloaders:
            downloader.download()


def main():
    models_dir = Path('./models')
    manager = ModelManager(models_dir)
    manager.download_all()


if __name__ == '__main__':
    main()
