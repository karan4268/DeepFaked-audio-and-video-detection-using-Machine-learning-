import os

def make_file_id(path: str, root: str):
    """
    STRICT CANONICAL KEY (VIDEO + AUDIO + FUSION SAFE)
    """

    rel = os.path.relpath(path, root)

    rel = rel.replace("\\", "/")

    # normalize dataset prefixes
    rel = rel.replace("FakeAVCeleb_v1.2/", "")
    rel = rel.replace("FakeAVCeleb/", "")

    rel = rel.lstrip("/")

    return rel