package com.example.jni;

import android.graphics.Bitmap;

public class geekeye_mobile_detection_1_0_0 {
	
	static {   
        System.loadLibrary("Geekeye_Mobile_Detection_1_0_0");   
    }
	
	//init
	/**
	 * @InputParam pImage : input ModelPath
     * 
	 * @OutputParam return[0]:res bin;
	 */
	public  native int DL_Init_Detection(String ModelPath);
	
	//image detection
	/**
	 * @InputParam pImage : input image
     * 
	 * @OutputParam return[0]:res bin;
	 * @OutputParam return[1]:times(ms);
	 * @OutputParam return[2]:num of res;
	 * @OutputParam return[2+2*N]:res(label,score*1000);
	 */
	public native int[] DL_Image_Detection( Bitmap pImage );
	
	//video detection
	public native int[] DL_Video_Detection( byte[] pImage, int previewWidth, int previewHeight );
	
}
