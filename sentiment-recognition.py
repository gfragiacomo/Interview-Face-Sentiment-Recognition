from moviepy import *
import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class VideoEmotionAnalyzer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.emotions_data = []
        self.timestamps = []
        self.frame_count = 0
        
    def analyze_frame(self, frame):
        """Process individual frame and detect emotions"""
        self.frame_count += 1
        frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        
        if self.frame_count % 5 == 0:  # Process every 5th frame
            print(f"Processing frame {self.frame_count}")
            try:
                # Analyze emotions in frame
                result = DeepFace.analyze(frame_bgr, 
                                        actions=['emotion'],
                                        enforce_detection=False,
                                        silent=True)
                
                if result:
                    emotions = result[0]['emotion']
                    self.emotions_data.append(emotions)
                    self.timestamps.append(self.frame_count / self.clip.fps)
                    
                    # Get face region and dominant emotion
                    face_box = result[0]['region']
                    dominant_emotion = result[0]['dominant_emotion']
                    emotion_score = emotions[dominant_emotion]
                    
                    # Draw rectangle around face
                    cv2.rectangle(frame_bgr,
                                (face_box['x'], face_box['y']),
                                (face_box['x'] + face_box['w'], 
                                 face_box['y'] + face_box['h']),
                                (0, 255, 0), 2)
                    
                    # Add emotion label
                    label = f"{dominant_emotion}: {emotion_score:.1f}%"
                    cv2.putText(frame_bgr,
                              label,
                              (face_box['x'], face_box['y'] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.9,
                              (0, 255, 0),
                              2)
                    
            except Exception as e:
                pass  # Skip frames where face detection fails
                    
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    def create_text_clips(self):
        """Create overlay text clips for emotions"""
        text_clips = []
        for time, emotion in zip(self.timestamps, self.emotions_data):
            dominant_emotion = max(emotion.items(), key=lambda x: x[1])[0]
            text_clip = TextClip(
                text=f"Dominant: {dominant_emotion}",
                font_size=30,
                color="white"
            ).set_position('top').set_duration(0.2).set_start(time)
            text_clips.append(text_clip)
        return text_clips
    
    def create_emotion_graph(self):
        """Generate and save emotion timeline graph"""
        df = pd.DataFrame(self.emotions_data, index=self.timestamps)
        plt.figure(figsize=(12, 6))
        for emotion in df.columns:
            plt.plot(df.index, df[emotion], label=emotion, alpha=0.7)
        plt.title('Emotion Timeline')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Confidence (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('emotion_graph.png')
        return df
    
    def process(self):
        """Main processing pipeline"""
        try:
            print("Loading video...")
            self.clip = VideoFileClip(self.video_path)
            
            print("Analyzing video...")
            processed_clip = self.clip.fl_image(self.analyze_frame)
            
            print("Adding annotations...")
            text_clips = self.create_text_clips()
            final_video = CompositeVideoClip([processed_clip] + text_clips)
            
            print("Saving processed video...")
            final_video.write_videofile(
                "analyzed_output.mp4",
                codec='libx264',
                audio_codec='aac'
            )
            
            print("Creating emotion graph...")
            results_df = self.create_emotion_graph()
            
            # Cleanup
            self.clip.close()
            final_video.close()
            
            print("Analysis complete!")
            return results_df
            
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            return None

def main():
    video_path = '/Users/utente/Movies/Sentiment Analysis/P1433721_Proxy.mov'
    analyzer = VideoEmotionAnalyzer(video_path)
    results = analyzer.process()
    
    if results is not None:
        print("\nAnalysis results saved to:")
        print("- analyzed_output.mp4 (processed video)")
        print("- emotion_graph.png (emotion timeline)")

if __name__ == "__main__":
    main()