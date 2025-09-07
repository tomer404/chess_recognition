import json
from pathlib import Path
from typing import Dict, Union
import pandas as pd
from detect import *


def read_chess_positions(
    dataroot: Union[str, Path],
    image: Union[int, str, Path],
    include_empty: bool = False
) -> Dict[str, str]:
    """
    Read the chess-piece positions for a given image.

    Args:
        dataroot: Folder containing annotations.json and the dataset files.
        image: Either the image_id (int) OR the image 'path' string
               as listed in annotations.json.
        include_empty: If True, include empty squares in the dict with value 'empty'.

    Returns:
        positions_dict: mapping like {"a8": "black rook", "e1": "white king", ...}
                        (empty squares omitted unless include_empty=True)
    """
    dataroot = Path(dataroot)
    ann_path = dataroot / "annotations.json"
    if not ann_path.is_file():
        raise FileNotFoundError(f"File '{ann_path}' doesn't exist.")

    with open(ann_path, "r") as f:
        annotations_file = json.load(f)

    # Tables
    annotations = pd.DataFrame(annotations_file["annotations"]["pieces"])
    categories = pd.DataFrame(annotations_file["categories"])
    images = pd.DataFrame(annotations_file["images"])

    # Resolve image_id
    if isinstance(image, int):
        img_rows = images[images["id"] == image]
    else:
        img_rows = images[images["path"] == str(image)]

    if len(img_rows) != 1:
        raise ValueError(
            f"Could not uniquely resolve image '{image}', matches found: {len(img_rows)}"
        )

    image_id = int(img_rows.iloc[0]["id"])

    # Category id -> name
    cat_id_to_name = {
        int(row["id"]): str(row["name"]).replace("_", " ").lower()
        for _, row in categories.iterrows()
    }

    # Filter annotations for this image
    anns_img = annotations[annotations["image_id"] == image_id].copy()

    positions: Dict[str, str] = {}
    for _, row in anns_img.iterrows():
        pos = str(row["chessboard_position"])  # e.g., "a8"
        cat_id = int(row["category_id"])
        name = cat_id_to_name.get(cat_id, f"id_{cat_id}")
        if include_empty or name != "empty":
            positions[pos] = name

    return positions

def num_of_incorrect_pieces(img_file_path):
    classes_map = {}
    pos_map = read_chess_positions(dataroot="", image= img_file_path, include_empty=False)
    detected_pos_map = main(img_file_path)
    cnt = 0
    for row in range(8):
        for col in {"a", "b", "c", "d", "e", "f", "g", "h"}:
            if pos_map.get(col+str(row)) != detected_pos_map.get(col+str(row)):
                cnt+=1
    return cnt

def pos_to_fen(pos_map):
    pass

# Testing the accuracy on the first folder
if __name__ == "__main__":
    accuracy = {}
    cnt = 0
    for i in range(103):
        accuracy[i] = num_of_incorrect_pieces(create_file_path(i, 0))
        if(accuracy[i]>0):
            cnt+=1
    print(f"The number of incorrect board positions is {cnt}")
    print(f"The number of incorrectly detected pieces in each photo: {accuracy}")