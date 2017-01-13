package com.example.geekeye_mobile_detection;
import com.example.geekeye_mobile_detection.R;

import android.support.v7.app.ActionBarActivity;
import android.util.DisplayMetrics;
import android.util.Log;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.List;

import com.example.jni.geekeye_mobile_detection_1_0_0;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Intent;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.Bitmap.CompressFormat;
import android.graphics.drawable.BitmapDrawable;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

public class MainActivity extends ActionBarActivity {
	
	private Button btn_getImg_camera, btn_getImg_local, btn_detect, btn_detect_string,btn_cropImage;
	private TextView mTxtResult;
	private ImageView mIVImage, mIVRes;
	//
    private String mImageFile;
    private Bitmap mCurrentImage = null;
    private Bitmap mResizeImage = null;
    private Bitmap mResImage = null;

    //
    private geekeye_mobile_detection_1_0_0 geekeye = null;
    private String modelPath = null;
    
    int wMonitor, hMonitor;
    boolean binImageView = false;

	@SuppressLint("SdCardPath")
	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);
		
		btn_getImg_camera = (Button) findViewById(R.id.btn_getImg_camera);
		btn_getImg_local = (Button) findViewById(R.id.btn_getImg_local);
		btn_detect = (Button) findViewById(R.id.btn_detect);
		btn_detect_string = (Button) findViewById(R.id.btn_detect_string);
		btn_cropImage = (Button) findViewById(R.id.btn_cropImage);
		mTxtResult = (TextView) findViewById(R.id.textView1);
		mIVImage = (ImageView) findViewById(R.id.imgview_load);
		mIVRes = (ImageView) findViewById(R.id.imgview_res);
		
		//imageview touch
		DisplayMetrics dm=new DisplayMetrics();//创建矩阵  
		getWindowManager().getDefaultDisplay().getMetrics(dm);
		wMonitor = dm.widthPixels;
		hMonitor = dm.heightPixels;
		
		//get model path
		modelPath = "/sdcard/geekeye-mobile-detection/models";
		mTxtResult.setText(modelPath);
		
		//load model
		geekeye = new geekeye_mobile_detection_1_0_0();
        int res = geekeye.DL_Init_Detection(modelPath);
        mTxtResult.setText("DL_Init_Detection end!!res:" + res );

		btn_getImg_camera.setOnClickListener(new Button.OnClickListener() {
			public void onClick(View v) {
				// 调用系统相机
				Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
				startActivityForResult(intent, 1);
			}
		});
		
		btn_getImg_local.setOnClickListener(new Button.OnClickListener() {
			public void onClick(View v) {
				Intent intent = new Intent(Intent.ACTION_PICK, null);
				intent.setDataAndType(MediaStore.Images.Media.EXTERNAL_CONTENT_URI,"image/*");
				startActivityForResult(intent, 2);
			}
		});	
		
		mIVRes.setOnClickListener(new ImageView.OnClickListener(){	
			public void onClick(View v) 
			{  
				float scaleWidth = 0;  
			    float scaleHeight = 0;  
			    
			    int width = mResImage.getWidth();  
			    int height = mResImage.getHeight();  
			    scaleWidth=((float)wMonitor)/width;  
			    scaleHeight=((float)hMonitor)/height;
			    
	        	Matrix matrix=new Matrix();  
	        	if( binImageView == true )
	        	{
	        		matrix.postScale(1.0f,1.0f); 
	        		binImageView = false;
	        	}
	        	else
	        	{
	        		matrix.postScale(scaleWidth,scaleHeight);
	        		binImageView = true;
	        	}
	   
	        	Bitmap newBitmap=Bitmap.createBitmap(mResImage, 0, 0, width, height, matrix, true);  
	        	mIVRes.setImageBitmap(newBitmap);    
			}
			
		});
	}
	
	@Override
	protected void onActivityResult(int requestCode, int resultCode, Intent data) {
		if (1 == requestCode) { // 系统相机返回处理
			if (resultCode == Activity.RESULT_OK) {
				Bitmap cameraBitmap = (Bitmap) data.getExtras().get("data");
                mResizeImage = ImageUtil.resizeImageMax(cameraBitmap, 512);
                mIVImage.setImageBitmap(mResizeImage);
                mTxtResult.setText("load image");

                // do job
                jobPhoto();
			}
			btn_getImg_camera.setClickable(true);
		} else if (2 == requestCode) {
			if (resultCode == Activity.RESULT_OK) {
                mImageFile = ImageUtil.getPhotoPath(MainActivity.this, data.getData());
                Toast.makeText(this, mImageFile, Toast.LENGTH_LONG).show();

                // origin image
                BitmapFactory.Options options = new BitmapFactory.Options();
                options.inPreferredConfig = Bitmap.Config.ARGB_8888;
                mCurrentImage = BitmapFactory.decodeFile(mImageFile,options);
                mResizeImage = ImageUtil.resizeImageMax(mCurrentImage, 512);
                mIVImage.setImageBitmap(mResizeImage);
                mTxtResult.setText("load image");

                // do job
                jobPhoto();
			}
		}
		super.onActivityResult(requestCode, resultCode, data);
	}
	
	//////////////////////////////////////////////
	private void jobPhoto()
	{
		//
		//mTxtResult.setText("jobPhoto in");
		long current = System.currentTimeMillis();
		int[] res = geekeye.DL_Image_Detection(mResizeImage);
		long performance = System.currentTimeMillis() - current;
		if( res.length == 3 )
			mTxtResult.setText("Res:"+res[0]+",t1:"+res[1]+"ms"+",t2:"+String.valueOf(performance)+"ms"+",size:"+res[2]);
		if( res[0] == 0 )
		{
			mResImage = getLoacalBitmap(modelPath+"/res.jpg");
			mIVRes.setImageBitmap(mResImage);
		}
		res = null;
	}
	
	private static Bitmap getLoacalBitmap(String url) 
	{
		try {
		     FileInputStream fis = new FileInputStream(url);
		     return BitmapFactory.decodeStream(fis);  ///把流转化为Bitmap图片        
		
        } 
		catch (FileNotFoundException e) {
             e.printStackTrace();
             return null;
		}
	}

	@Override
	public boolean onCreateOptionsMenu(Menu menu) {
		// Inflate the menu; this adds items to the action bar if it is present.
		getMenuInflater().inflate(R.menu.main, menu);
		return true;
	}

	@Override
	public boolean onOptionsItemSelected(MenuItem item) {
		// Handle action bar item clicks here. The action bar will
		// automatically handle clicks on the Home/Up button, so long
		// as you specify a parent activity in AndroidManifest.xml.
		int id = item.getItemId();
		if (id == R.id.action_settings) {
			return true;
		}
		return super.onOptionsItemSelected(item);
	}
}
