# omr_evaluator.py
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd
import json
import tempfile
import os
from scipy import ndimage

class OMREvaluator:
    def _init_(self):
        # Configuration - would be loaded from a config file in production
        self.sheet_config = {
            'width': 800,
            'height': 1100,
            'bubble_radius': 15,
            'bubble_spacing_x': 60,
            'bubble_spacing_y': 50,
            'bubbles_per_question': 5,
            'questions_per_subject': 20,
            'num_subjects': 5,
            'first_bubble_x': 150,
            'first_bubble_y': 300,
            'version_markers': [
                {'x': 50, 'y': 50, 'value': 'A'},
                {'x': 750, 'y': 50, 'value': 'B'},
                {'x': 50, 'y': 1050, 'value': 'C'},
                {'x': 750, 'y': 1050, 'value': 'D'}
            ]
        }
        
        # Sample answer keys for different versions
        self.answer_keys = {
            'A': {
                'subject1': [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
                'subject2': [1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0],
                'subject3': [2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1],
                'subject4': [3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2],
                'subject5': [4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3]
            },
            'B': {
                'subject1': [1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0],
                'subject2': [2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1],
                'subject3': [3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2],
                'subject4': [4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3],
                'subject5': [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
            },
            'C': {
                'subject1': [2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1],
                'subject2': [3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2],
                'subject3': [4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3],
                'subject4': [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
                'subject5': [1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0]
            },
            'D': {
                'subject1': [3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2],
                'subject2': [4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3],
                'subject3': [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
                'subject4': [1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0],
                'subject5': [2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1]
            }
        }
    
    def preprocess_image(self, image):
        """Preprocess the image for OMR evaluation"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        return thresh
    
    def detect_sheet_version(self, image):
        """Detect the sheet version based on markers"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        version = None
        max_matches = 0
        
        for marker in self.sheet_config['version_markers']:
            # Create a template for the marker (simple circle)
            template = np.zeros((30, 30), dtype=np.uint8)
            cv2.circle(template, (15, 15), 12, 255, -1)
            
            # Extract ROI around the marker position
            x, y = marker['x'], marker['y']
            roi = gray[max(0, y-20):min(gray.shape[0], y+20), 
                      max(0, x-20):min(gray.shape[1], x+20)]
            
            if roi.size == 0:
                continue
                
            # Resize ROI to match template size if needed
            if roi.shape[0] != 30 or roi.shape[1] != 30:
                roi = cv2.resize(roi, (30, 30))
            
            # Match template
            result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            # If this marker has a strong match, consider it
            if max_val > 0.7 and max_val > max_matches:
                max_matches = max_val
                version = marker['value']
        
        return version if version else 'A'  # Default to version A if detection fails
    
    def correct_perspective(self, image):
        """Correct perspective distortion of the OMR sheet"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find contours
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Approximate the contour with a polygon
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # If we found a quadrilateral
            if len(approx) == 4:
                # Order points for perspective transform
                rect = self.order_points(approx.reshape(4, 2))
                
                # Calculate the width and height of the new image
                width = self.sheet_config['width']
                height = self.sheet_config['height']
                
                # Destination points for the transform
                dst = np.array([
                    [0, 0],
                    [width - 1, 0],
                    [width - 1, height - 1],
                    [0, height - 1]
                ], dtype="float32")
                
                # Compute the perspective transform matrix and apply it
                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(image, M, (width, height))
                
                return warped
        
        # If no quadrilateral found, return original image
        return image
    
    def order_points(self, pts):
        """Order points in clockwise order starting from top-left"""
        rect = np.zeros((4, 2), dtype="float32")
        
        # The top-left point will have the smallest sum
        # The bottom-right point will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # The top-right point will have the smallest difference
        # The bottom-left point will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def detect_bubbles(self, image):
        """Detect and evaluate bubbles on the OMR sheet"""
        # Preprocess the image
        processed = self.preprocess_image(image)
        
        # Detect sheet version
        version = self.detect_sheet_version(image)
        
        # Get answer key for this version
        answer_key = self.answer_keys[version]
        
        # Initialize results
        responses = {}
        scores = {}
        
        # Evaluate each subject
        for subject_idx in range(1, self.sheet_config['num_subjects'] + 1):
            subject_key = f'subject{subject_idx}'
            responses[subject_key] = []
            scores[subject_key] = 0
            
            # Evaluate each question in the subject
            for question_idx in range(self.sheet_config['questions_per_subject']):
                # Calculate bubble positions for this question
                y_pos = self.sheet_config['first_bubble_y'] + question_idx * self.sheet_config['bubble_spacing_y']
                bubbles = []
                
                for option_idx in range(self.sheet_config['bubbles_per_question']):
                    x_pos = self.sheet_config['first_bubble_x'] + option_idx * self.sheet_config['bubble_spacing_x']
                    
                    # Extract the bubble region
                    bubble_roi = processed[
                        max(0, y_pos - self.sheet_config['bubble_radius']):min(processed.shape[0], y_pos + self.sheet_config['bubble_radius']),
                        max(0, x_pos - self.sheet_config['bubble_radius']):min(processed.shape[1], x_pos + self.sheet_config['bubble_radius'])
                    ]
                    
                    # Count non-zero pixels (marked area)
                    if bubble_roi.size > 0:
                        bubble_value = cv2.countNonZero(bubble_roi)
                        bubbles.append(bubble_value)
                    else:
                        bubbles.append(0)
                
                # Determine if any bubble is marked
                if bubbles:
                    # Find the bubble with the maximum marks
                    max_bubble = np.argmax(bubbles)
                    max_value = bubbles[max_bubble]
                    
                    # Apply threshold to determine if bubble is marked
                    threshold = 0.4 * np.pi * (self.sheet_config['bubble_radius'] ** 2)
                    
                    if max_value > threshold:
                        responses[subject_key].append(max_bubble)
                        
                        # Check if answer is correct
                        if max_bubble == answer_key[subject_key][question_idx]:
                            scores[subject_key] += 1
                    else:
                        responses[subject_key].append(-1)  # No answer
                else:
                    responses[subject_key].append(-1)  # No answer
        
        # Calculate total score
        total_score = sum(scores.values())
        
        return {
            'version': version,
            'responses': responses,
            'scores': scores,
            'total_score': total_score,
            'processed_image': processed
        }
    
    def evaluate_omr_sheet(self, image_path):
        """Main function to evaluate an OMR sheet"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not load image")
        
        # Correct perspective
        corrected_image = self.correct_perspective(image)
        
        # Detect bubbles and evaluate
        result = self.detect_bubbles(corrected_image)
        
        # Add original and corrected images to result for display
        result['original_image'] = image
        result['corrected_image'] = corrected_image
        
        return result

# Streamlit Web Application
def main():
    st.set_page_config(page_title="OMR Evaluation System", layout="wide")
    st.title("Automated OMR Evaluation System")
    
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'evaluator' not in st.session_state:
        st.session_state.evaluator = OMREvaluator()
    
    # Sidebar for upload and actions
    with st.sidebar:
        st.header("Upload OMR Sheets")
        uploaded_files = st.file_uploader(
            "Choose OMR sheet images", 
            type=['jpg', 'jpeg', 'png'], 
            accept_multiple_files=True
        )
        
        if st.button("Evaluate All Sheets"):
            if uploaded_files:
                with st.spinner("Processing sheets..."):
                    for uploaded_file in uploaded_files:
                        # Save uploaded file to temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        try:
                            # Evaluate the OMR sheet
                            result = st.session_state.evaluator.evaluate_omr_sheet(tmp_path)
                            result['filename'] = uploaded_file.name
                            st.session_state.results.append(result)
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                        finally:
                            # Clean up temporary file
                            os.unlink(tmp_path)
                
                st.success(f"Processed {len(uploaded_files)} sheets!")
            else:
                st.warning("Please upload at least one OMR sheet image.")
        
        if st.button("Clear Results"):
            st.session_state.results = []
    
    # Main content area
    if st.session_state.results:
        st.header("Evaluation Results")
        
        # Display summary statistics
        total_sheets = len(st.session_state.results)
        avg_score = np.mean([r['total_score'] for r in st.session_state.results])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Sheets", total_sheets)
        col2.metric("Average Score", f"{avg_score:.2f}/100")
        col3.metric("Processing Rate", "100%")
        
        # Detailed results for each sheet
        for i, result in enumerate(st.session_state.results):
            with st.expander(f"Sheet {i+1}: {result['filename']} (Version {result['version']}, Score: {result['total_score']}/100)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original Image")
                    st.image(cv2.cvtColor(result['original_image'], cv2.COLOR_BGR2RGB), 
                            caption="Uploaded Image", use_column_width=True)
                
                with col2:
                    st.subheader("Processed Image")
                    st.image(result['processed_image'], 
                            caption="After Preprocessing", use_column_width=True, clamp=True)
                
                # Display scores by subject
                st.subheader("Subject-wise Scores")
                subject_scores = result['scores']
                subject_df = pd.DataFrame({
                    'Subject': list(subject_scores.keys()),
                    'Score': [f"{score}/20" for score in subject_scores.values()]
                })
                st.table(subject_df)
                
                # Display detailed responses
                st.subheader("Detailed Responses")
                responses = result['responses']
                response_data = []
                
                for subject, answers in responses.items():
                    for j, answer in enumerate(answers):
                        response_data.append({
                            'Subject': subject,
                            'Question': j+1,
                            'Answer': chr(65 + answer) if answer != -1 else 'No Answer',
                            'Correct': 'Yes' if answer == st.session_state.evaluator.answer_keys[result['version']][subject][j] else 'No'
                        })
                
                response_df = pd.DataFrame(response_data)
                st.dataframe(response_df, hide_index=True, use_container_width=True)
                
                # Download button for results
                json_result = json.dumps({
                    'filename': result['filename'],
                    'version': result['version'],
                    'scores': result['scores'],
                    'total_score': result['total_score'],
                    'responses': result['responses']
                }, indent=2)
                
                st.download_button(
                    label=f"Download Results for Sheet {i+1}",
                    data=json_result,
                    file_name=f"results_{result['filename']}.json",
                    mime="application/json",
                    key=f"download_{i}"
                )
        
        # Export all results as CSV
        if st.button("Export All Results as CSV"):
            export_data = []
            for result in st.session_state.results:
                row = {
                    'Filename': result['filename'],
                    'Version': result['version'],
                    'Total Score': result['total_score']
                }
                
                # Add subject scores
                for subject, score in result['scores'].items():
                    row[subject] = score
                
                export_data.append(row)
            
            export_df = pd.DataFrame(export_data)
            csv = export_df.to_csv(index=False)
            
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="omr_results.csv",
                mime="text/csv"
            )
    else:
        # Show instructions if no results
        st.info("""
        ## Welcome to the OMR Evaluation System
        
        *Instructions:*
        1. Upload OMR sheet images using the sidebar
        2. Click 'Evaluate All Sheets' to process them
        3. View results and export as needed
        
        *Supported formats:* JPG, JPEG, PNG
        *Sheet requirements:* 
        - Clear image of the OMR sheet
        - Good lighting conditions
        - Sheet should fill most of the image frame
        """)
        
        # Placeholder for sample image
        st.image("https://via.placeholder.com/600x400?text=Upload+OMR+Sheets+to+Begin", 
                use_column_width=True)

if __name__ == "_main_":
    main()
