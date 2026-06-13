"""
Ollama-compatible API responses for Oprel SDK
"""

from typing import Dict, List, Any, Optional
from datetime import datetime


class Message:
    """Chat message container"""
    
    def __init__(self, role: str, content: str, images: Optional[List[str]] = None):
        self.role = role
        self.content = content
        self.images = images or []
    
    def __getitem__(self, key: str):
        """Allow dict-like access"""
        return getattr(self, key)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = {"role": self.role, "content": self.content}
        if self.images:
            data["images"] = self.images
        return data


class ChatResponse:
    """Response from chat endpoint"""
    
    def __init__(
        self,
        model: str,
        message: Message,
        created_at: Optional[str] = None,
        done: bool = True,
        total_duration: Optional[int] = None,
        load_duration: Optional[int] = None,
        prompt_eval_count: Optional[int] = None,
        eval_count: Optional[int] = None,
        eval_duration: Optional[int] = None,
    ):
        self.model = model
        self.message = message
        self.created_at = created_at or datetime.utcnow().isoformat()
        self.done = done
        self.total_duration = total_duration
        self.load_duration = load_duration
        self.prompt_eval_count = prompt_eval_count
        self.eval_count = eval_count
        self.eval_duration = eval_duration
    
    def __getitem__(self, key: str):
        """Allow dict-like access"""
        return getattr(self, key)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "model": self.model,
            "message": self.message.to_dict(),
            "created_at": self.created_at,
            "done": self.done,
            "total_duration": self.total_duration,
            "load_duration": self.load_duration,
            "prompt_eval_count": self.prompt_eval_count,
            "eval_count": self.eval_count,
            "eval_duration": self.eval_duration,
        }


class GenerateResponse:
    """Response from generate endpoint"""
    
    def __init__(
        self,
        model: str,
        response: str,
        created_at: Optional[str] = None,
        done: bool = True,
        context: Optional[List[int]] = None,
        total_duration: Optional[int] = None,
        load_duration: Optional[int] = None,
        prompt_eval_count: Optional[int] = None,
        eval_count: Optional[int] = None,
        eval_duration: Optional[int] = None,
    ):
        self.model = model
        self.response = response
        self.created_at = created_at or datetime.utcnow().isoformat()
        self.done = done
        self.context = context or []
        self.total_duration = total_duration
        self.load_duration = load_duration
        self.prompt_eval_count = prompt_eval_count
        self.eval_count = eval_count
        self.eval_duration = eval_duration
    
    def __getitem__(self, key: str):
        """Allow dict-like access"""
        return getattr(self, key)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "model": self.model,
            "response": self.response,
            "created_at": self.created_at,
            "done": self.done,
            "context": self.context,
            "total_duration": self.total_duration,
            "load_duration": self.load_duration,
            "prompt_eval_count": self.prompt_eval_count,
            "eval_count": self.eval_count,
            "eval_duration": self.eval_duration,
        }


class ImageResponse:
    """Response from image generation endpoint"""

    def __init__(
        self,
        created: int,
        data: List[Dict[str, Any]],
    ):
        self.created = created
        self.data = data

    def __getitem__(self, key: str):
        """Allow dict-like access"""
        return getattr(self, key)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "created": self.created,
            "data": self.data,
        }


class ModelInfo:
    """Model information response"""
    
    def __init__(
        self,
        name: str,
        modified_at: str,
        size: int,
        digest: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.modified_at = modified_at
        self.size = size
        self.digest = digest
        self.details = details or {}
    
    def __getitem__(self, key: str):
        """Allow dict-like access"""
        return getattr(self, key)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "modified_at": self.modified_at,
            "size": self.size,
            "digest": self.digest,
            "details": self.details,
        }


class ListResponse:
    """Response from list endpoint"""
    
    def __init__(self, models: List[ModelInfo]):
        self.models = models
    
    def __getitem__(self, key: str):
        """Allow dict-like access"""
        return getattr(self, key)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "models": [m.to_dict() for m in self.models]
        }


class ShowResponse:
    """Response from show endpoint"""
    
    def __init__(
        self,
        modelfile: str,
        parameters: str,
        template: str,
        details: Dict[str, Any],
    ):
        self.modelfile = modelfile
        self.parameters = parameters
        self.template = template
        self.details = details
    
    def __getitem__(self, key: str):
        """Allow dict-like access"""
        return getattr(self, key)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "modelfile": self.modelfile,
            "parameters": self.parameters,
            "template": self.template,
            "details": self.details,
        }
