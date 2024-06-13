from scripts.fop_recognition import FOPRecognition

if __name__ == '__main__':
    
    save_image = input('Do you want save crop image? \n Please enter - y/n:')
    if save_image == 'y':
        crop_dir_name = input('Enter dir for crop image:')
    else:
        save_image = None
        crop_dir_name = ''

    write_video = input('Do you want write and show video with detection? \n Please enter - y/n:')
    if write_video == 'y':
        video_write_dir_name = input('Enter dir for video:')
    else:
        write_video = None
        video_write_dir_name = ''

    results_file = input('Do you want save file with results? \n Please enter - y/n:')
    if results_file == 'y':
        results_file_dir_name = input('Enter dir results file:')
    else:
        results_file = None
        results_file_dir_name = ''

    video_path = input('Please enter path for video file -')

    app = FOPRecognition(
        save_image = save_image,
        write_video = write_video,
        results_file = results_file
    )

    results = app.face_recogniton(
        video_path = video_path,
        crop_dir_name = crop_dir_name,
        video_write_dir_name = video_write_dir_name,
        results_file_dir_name = results_file_dir_name
    )
    print(results)