from moviepy import *
import json

def convert_timestamp(timestamp):
    """Convert timestamp like '0:01:27.4' to seconds"""
    # Remove trailing zeros if present
    timestamp = timestamp.split('.')[0] + '.' + timestamp.split('.')[1][:1]
    h, m, s = timestamp.split(':')
    return float(h) * 3600 + float(m) * 60 + float(s)

def create_labeled_video(video_path, json_path):
    print(f"Processing video: {video_path}")
    
    # Load video and json
    video = VideoFileClip(video_path)
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract emotions data from insights
    emotions_data = data['videos'][0]['insights']['emotions']
    text_clips = []
    
    print("Creating emotion overlays...")
    
    # Create clips for each emotion segment
    for emotion in emotions_data:
        emotion_type = emotion['type']
        
        for instance in emotion['instances']:
            start_time = convert_timestamp(instance['adjustedStart'])
            end_time = convert_timestamp(instance['adjustedEnd'])
            confidence = instance['confidence'] * 100  # Convert to percentage
            
            # Create emotion label with font specified
            emotion_txt = (TextClip(
                txt=f"{emotion_type}: {confidence:.1f}%",
                font='Arial',  # Specify font
                fontsize=50,
                color='white',
                bg_color='rgba(0,0,0,0.5)',
                method='caption')  # Use caption method for better rendering
                .set_start(start_time)
                .set_duration(end_time - start_time)
                .set_position(('center', 'bottom')))
            text_clips.append(emotion_txt)
            
            # Add sentiment data if available
            for sentiment in data['videos'][0]['insights']['sentiments']:
                for sent_instance in sentiment['instances']:
                    if (convert_timestamp(sent_instance['adjustedStart']) <= start_time and 
                        convert_timestamp(sent_instance['adjustedEnd']) >= end_time):
                        sentiment_txt = (TextClip(
                            txt=f"Sentiment: {sentiment['sentimentType']}",
                            font='Arial',  # Specify font
                            fontsize=30,
                            color='white',
                            bg_color='rgba(0,0,0,0.5)',
                            method='caption')  # Use caption method
                            .set_start(start_time)
                            .set_duration(end_time - start_time)
                            .set_position(('center', 50)))
                        text_clips.append(sentiment_txt)
    
    # Combine video with text overlays
    print("Combining video with labels...")
    final_video = CompositeVideoClip([video] + text_clips)
    
    # Write output
    output_path = video_path.rsplit('.', 1)[0] + '_with_emotions.mp4'
    print(f"Writing output to: {output_path}")
    
    final_video.write_videofile(
        output_path,
        codec='libx264',
        audio_codec='aac'
    )
    
    # Cleanup
    video.close()
    final_video.close()
    print("Processing complete!")

def main():
    from tkinter import Tk, filedialog
    
    root = Tk()
    root.withdraw()
    
    print("=== Video Emotion Labeling Tool ===\n")
    
    # Select video file
    print("Please select your video file:")
    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video files", "*.mp4 *.mov *.avi"), ("All files", "*.*")]
    )
    if not video_path:
        print("No video file selected. Exiting.")
        return
        
    # Select JSON file
    print("\nPlease select your Video Indexer JSON file:")
    json_path = filedialog.askopenfilename(
        title="Select JSON File",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
    )
    if not json_path:
        print("No JSON file selected. Exiting.")
        return
    
    try:
        create_labeled_video(video_path, json_path)
    except Exception as e:
        print(f"\nError processing video: {str(e)}")
        return

if __name__ == "__main__":
    main()