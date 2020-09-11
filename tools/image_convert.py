import shutil
import cv2 as cv
from pathlib import Path


def do_convert(in_dir, origin=".bmp", target=".png", color=1):
    in_dir = Path(in_dir)
    out_dir = in_dir.name + "_cvt"
    out_dir = in_dir.parent / out_dir

    num_image, num_other = 0, 0
    shutil.rmtree(out_dir, ignore_errors=True)
    for cur_path in Path(in_dir).glob("**/*"):
        out_file = out_dir / cur_path.relative_to(in_dir)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        if cur_path.suffix == origin:
            num_image += 1
            im = cv.imread(cur_path.as_posix(), color)
            cv.imwrite(out_file.with_suffix(target).as_posix(), im)
        elif cur_path.is_file():
            num_other += 1
            shutil.copyfile(cur_path, out_file)
    return out_dir.as_posix(), num_image, num_other
