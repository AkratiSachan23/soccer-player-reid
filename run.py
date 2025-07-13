from player_reid.pipeline import PlayerReIDPipeline

if __name__ == "__main__":
    pipeline = PlayerReIDPipeline("player_reid\configs\default.yaml")
    pipeline.process_video()
    print("Processing complete. Output saved to data/output/reid_output_video.mp4")