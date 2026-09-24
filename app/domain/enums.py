from enum import Enum


class BeatType(str, Enum):
    HOOK = "HOOK"
    KNOWLEDGE = "KNOWLEDGE"
    EXAMPLE = "EXAMPLE"
    TRANSITION = "TRANSITION"
    SUMMARY = "SUMMARY"


class VisualType(str, Enum):
    STOCK_VIDEO = "STOCK_VIDEO"
    AI_VIDEO = "AI_VIDEO"
    AI_IMAGE = "AI_IMAGE"
    DIAGRAM = "DIAGRAM"
    SOURCE_ASSET = "SOURCE_ASSET"
    USER_ASSET = "USER_ASSET"


class StoryboardSnapshotState(str, Enum):
    DRAFT = "DRAFT"
    APPROVED = "APPROVED"
