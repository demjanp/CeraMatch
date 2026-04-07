from ceramatch import __version__
import os
import re
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ftemplate = os.path.join(SCRIPT_DIR, "ceramatch_installer.tpl")


def get_product_version(version):
    parts = [int(part) for part in re.findall(r"\d+", version)]
    parts = (parts + [0, 0, 0, 0])[:4]
    return ".".join(str(part) for part in parts)


def get_filename_version(version):
    return re.sub(r"[^0-9A-Za-z._-]+", "_", version)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 1:
        raise Exception("Invalid number of arguments.\nCorrect use: make_nsi.py <CeraMatch root dir>")

    root_path, = args
    root_path = os.path.normpath(os.path.abspath(root_path))

    installer_path = os.path.join(root_path, "dist")
    os.makedirs(installer_path, exist_ok=True)

    with open(ftemplate, "r", encoding="utf-8-sig") as handle:
        text = handle.read()

    text = text % dict(
        display_version=__version__,
        product_version=get_product_version(__version__),
        filename="ceramatch_%s_setup.exe" % get_filename_version(__version__),
        root_path=root_path,
        installer_path=installer_path,
    )

    output_path = os.path.join(installer_path, "ceramatch_installer.nsi")
    with open(output_path, "w", encoding="utf-8-sig") as handle:
        handle.write(text)
