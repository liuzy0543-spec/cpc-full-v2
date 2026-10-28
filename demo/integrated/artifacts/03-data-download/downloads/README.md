# 真实下载的对象（已裁剪）

这一轮 `data-download` 真的从 S3（unsigned）下载了两个照明校正文件：

- `images/2021_04_26_Batch1/illum/BR00117035/BR00117035_IllumAGP.npy`
- `images/2021_04_26_Batch1/illum/BR00117035/BR00117035_IllumBrightfield.npy`

每个约 4.45 MB，为了不让仓库背这个体积，**文件本身没有保留**，
下载记录见同目录 `download_manifest.json`（含对象键、字节数、sha256）。
重跑 `python verify_all.py --skills` 会把它们再下一次。
