def show_HDMI_capture(self):
    """
    HDMI를 display하는 함수
    """
    end = time.time()
    print(f"Display image load time : {end-self.start: .5f} sec")
    self.start = time.time()
    ret, frame = self.cap_video.read()
    if ret:
        #cv2.imwrite(f'./capture_with_UI/Frame_{self.frame_count}.png', frame)
        #image preprocessing 부분
        # 중앙 정렬된 영상일 경우 여백 80,130,340,340
        crop_img = image_crop_3ch(frame, self.ROI_x1, self.ROI_y1, self.ROI_x2, self.ROI_y2) 
        resize_img = image_scaling(crop_img, (500,460), 1)
        #cv2.imwrite(f'{self.temp_folder}Frame_{self.frame_count}.png', resize_img)
        #print(f'C:\\Users\\USER_1\\Desktop\\code_review\\capture\\Frame_{frame_count}.png')
        self.frame_count += 1
        #time.sleep(1/fps)

    self.pyqt_display_func(resize_img, self.labelImage)
        
    #img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)
    #height, width, channel = img.shape
    #bytesPerLine = 3*width
    #qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)

    #pixmap = QPixmap(image_path)
        
    self.labelMeasureROSC.setText(f'Frame_{self.frame_count}.png') # 이미지 이름 업데이트
    self.current_image_index += 1