import hashlib
import json
import os
import shutil
from typing import Dict, Optional, Tuple
from domain.entities import Artifact


class ArtifactStore:
    def __init__(self, base_dir: str = "data/artifacts"):
        self.base_dir = base_dir
        self.metadata_file = os.path.join(base_dir, "metadata.json")
        os.makedirs(self.base_dir, exist_ok=True)
        self._registry: Dict[str, Artifact] = self._load_registry()

    def _load_registry(self) -> Dict[str, Artifact]:
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return {k: Artifact(**v) for k, v in data.items()}
            except Exception as e:
                print(f"Warning: could not load artifact registry: {e}")
        return {}

    def _save_registry(self):
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            data = {k: v.model_dump() for k, v in self._registry.items()}
            json.dump(data, f, indent=2)

    def _compute_sha256(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def save_bytes(
        self,
        content: bytes,
        file_type: str,
        transformation_name: str = "raw",
        settings: Optional[dict] = None,
        source_artifact_id: Optional[str] = None,
    ) -> Artifact:
        sha256_hash = self._compute_sha256(content)
        ext = file_type.lstrip(".").lower()
        if not ext:
            ext = "bin"
        
        rel_dir = os.path.join(sha256_hash[:2])
        target_dir = os.path.join(self.base_dir, rel_dir)
        os.makedirs(target_dir, exist_ok=True)
        
        file_name = f"{sha256_hash}.{ext}"
        file_path = os.path.join(target_dir, file_name)
        
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(content)
                
        artifact = Artifact(
            sha256_hash=sha256_hash,
            file_path=os.path.abspath(file_path),
            file_type=ext,
            source_artifact_id=source_artifact_id,
            transformation_name=transformation_name,
            settings=settings or {},
        )
        
        self._registry[artifact.artifact_id] = artifact
        self._save_registry()
        return artifact

    def save_file(
        self,
        source_file_path: str,
        transformation_name: str = "raw",
        settings: Optional[dict] = None,
        source_artifact_id: Optional[str] = None,
    ) -> Artifact:
        with open(source_file_path, "rb") as f:
            content = f.read()
        ext = os.path.splitext(source_file_path)[1].lstrip(".").lower() or "bin"
        return self.save_bytes(
            content=content,
            file_type=ext,
            transformation_name=transformation_name,
            settings=settings,
            source_artifact_id=source_artifact_id,
        )

    def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        return self._registry.get(artifact_id)

    def get_artifact_path(self, artifact_id: str) -> Optional[str]:
        art = self.get_artifact(artifact_id)
        if art and os.path.exists(art.file_path):
            return art.file_path
        return None

    def get_artifact_bytes(self, artifact_id: str) -> Optional[bytes]:
        path = self.get_artifact_path(artifact_id)
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                return f.read()
        return None


# Global singleton instance for easy import
artifact_store = ArtifactStore()
