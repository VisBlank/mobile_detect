package com.example.geekeye_mobile_detection;

import android.app.Activity;
import android.os.Environment;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

public class FileUtil {

    /**
     *
     * 鍒濆鍖�
     *
     * */
    public static String ResaveRaw(int rawId, String fileName, Activity activity) {
        String datDir = null;
        if (Environment.getExternalStorageState().equals(
                Environment.MEDIA_MOUNTED)) {// 浼樺厛淇濆瓨鍒癝D鍗′腑
            datDir = Environment.getExternalStorageDirectory()
                    .getAbsolutePath() + File.separator + "geekeye";
        } else {// 濡傛灉SD鍗′笉瀛樺湪锛屽氨淇濆瓨鍒版湰搴旂敤鐨勭洰褰曚笅
            datDir = activity.getApplicationContext().getFilesDir().getAbsolutePath()
                    + File.separator + "geekeye";
        }
        //datDir = "/storage/sdcard0/geekeye";
        String datFilePath = datDir + "/" + fileName;

        try {
            File datFs = new File(datFilePath);
            if (datFs.exists()) {
                return datFilePath;
            }
            //
            File dirFs = new File(datDir);
            if (!dirFs.exists()) {
                dirFs.mkdir();
                //Toast.makeText(this, datDir, Toast.LENGTH_LONG).show();
            }
            //
            InputStream is = activity.getResources().openRawResource(rawId);
            datFs = SaveFile(is, datFs);
            is.close();
        } catch (Exception e) {
            //Toast.makeText(this, e.getMessage().toString(), Toast.LENGTH_LONG)
            //        .show();
            return null;
        }

        return datFilePath;
    }

    /**
     * 灏咺nputStream閲岄潰鐨勬暟鎹啓鍏ュ埌SD鍗′腑
     */
    public static File SaveFile(InputStream inputStream, File file) {
        // 瀹氫箟缂撳瓨鍖哄ぇ灏�
        int FILESIZE = 4 * 1024;
        // 瀹氫箟涓�涓緭鍑烘祦锛岀敤鏉ュ啓鏁版嵁
        OutputStream outputStream = null;
        try {
            // 鏋勯�犱竴涓柊鐨勬枃浠惰緭鍑烘祦鍐欏叆鏂囦欢
            outputStream = new FileOutputStream(file);
            // 鍒涘缓涓�涓紦瀛樺尯
            byte[] buffer = new byte[FILESIZE];
            // 浠庤緭鍏ユ祦涓鍙栫殑鏁版嵁瀛樺叆缂撳瓨鍖�
            while ((inputStream.read(buffer)) != -1) {
                // 灏嗙紦瀛樺尯鐨勬暟鎹啓鍏ヨ緭鍑烘祦
                outputStream.write(buffer);
            }
            // 鍒锋柊娴侊紝娓呯┖缂撳啿鍖烘暟鎹�
            outputStream.flush();
        } catch (FileNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        } finally {
            try {
                // 鍏抽棴娴�
                outputStream.close();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        return file;
    }

    /*

     */
    public  static List<String> GetSynset(String synsetFile)
    {
        File file = new File(synsetFile);
        BufferedReader reader = null;
        String line = null;
        StringBuffer buffer = new StringBuffer();
        List<String> list = new ArrayList<String>();
        try {
            reader = new BufferedReader(new FileReader(file));
            while((line = reader.readLine()) != null){
                list.add(line);
            }
            reader.close();
            return list;
        } catch (FileNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
            return null;
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
            return null;
        }
    }

}
