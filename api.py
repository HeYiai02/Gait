import os
import shutil
import json
import uvicorn
from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import subprocess
from pose_extractor import RobustPoseExtractor
from predict_video import predict_gait_risk 

app = FastAPI(title="Gait Analysis API", description="步态识别与体检服务")


def convert_to_web_mp4(source_path: str, target_path: str):
    """
    【核心转码函数】: 将任意视频强行转换为前端 H5 完美支持的 H.264 编码
    """
    # -y: 自动覆盖同名文件
    # -vcodec libx264: 强制使用 H.264 编码
    # -pix_fmt yuv420p: 强制像素格式（非常关键，否则部分 Chrome 仍会黑屏）
    ffmpeg_exe = "E:/code/Env/ffmpeg-gpl-shared/bin/ffmpeg.exe"
    cmd = [
        ffmpeg_exe, "-y",
        "-i", source_path,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "faststart",
        target_path
    ]
    print("source_path:"+source_path)
    print("target_path:"+target_path)

    try:
        # check=True 会在转码失败时直接抛出异常
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        raise Exception(f"FFmpeg 转码失败: {e.stderr.decode('utf-8', errors='ignore')}")


@app.post("/analyze_gait/")
async def analyze_gait(video: UploadFile = File(...),output_video_path_prefix=""):
    """
    接收 Java 传来的视频，处理后返回诊断结果和视频路径
    """
    # 1. 准备必要的目录
    os.makedirs("data/temp", exist_ok=True)
    os.makedirs("data/output", exist_ok=True)
    
    # 2. 保存 Java 通过 HTTP 传过来的视频流
    temp_input_path = os.path.join("data/temp", video.filename)
    with open(temp_input_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
        
    # 构建输出路径 (提取纯文件名，不要后缀)
    base_name = os.path.splitext(video.filename)[0]
    output_csv_path = os.path.abspath(f"data/output/{base_name}_pose.csv")
    # output_video_path = os.path.abspath(f"data/output/{base_name}_annotated.mp4")
    output_video_name = base_name+"_annotated.mp4"
    output_video_path_temp = os.path.abspath("E:/code/lunwen/ruoyi/uploadPath/upload/"+base_name+"_raw.mp4")
    output_video_path = os.path.abspath("E:/code/lunwen/ruoyi/uploadPath/upload/"+base_name+"_annotated.mp4")
    
    try:
        # 3. 第一步：调用你的骨架提取器处理视频，生成 csv 数据和 画好骨架的新视频
        extractor = RobustPoseExtractor(conf_thresh=0.35, smooth_factor=0.4, enable_gmc=True, frame_stride=2)
        extractor.extract(temp_input_path, output_csv_path, output_video_path_temp)
        convert_to_web_mp4(output_video_path_temp,output_video_path)
        # 4. 第二步：执行医学指标打分
        result = predict_gait_risk(output_csv_path)
        if result is None:
            result = {}
            
        # 5. 补充成功状态和生成的视频绝对路径
        result["status"] = "success"
        result["annotatedVideoPath"] = output_video_path
        result["annotatedVideoName"] = output_video_name
        
        # FastAPI 会自动把这个字典转成纯净的 JSON 返回给 Java！
        return result
        
    except Exception as e:
        print(str(e))
        return {"status": "error", "message": str(e)}
        
    finally:
        # 6. 【扫尾工作】：处理完毕后，删除用来分析的原视频，释放空间
        # 画好骨架的新视频 output_video_path 会保留在硬盘上
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if os.path.exists(output_video_path_temp):
            os.remove(output_video_path_temp)
if __name__ == "__main__":
    # 启动命令：直接在终端运行 python api.py 即可启动服务
    uvicorn.run(app, host="127.0.0.1", port=8000)