package com.example.carscratchdetector.view;

import android.content.Intent;


import android.database.Cursor;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.ActionBar;
import androidx.appcompat.app.AppCompatActivity;

import com.bumptech.glide.Glide;

import com.example.carscratchdetector.R;
import com.example.carscratchdetector.controller.ApiController;
import com.example.carscratchdetector.controller.impl.ApiControllerImpl;
import com.example.carscratchdetector.dto.MaskRequestDto;
import com.example.carscratchdetector.dto.MaskResponseDto;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.firebase.FirebaseApp;
import com.google.firebase.FirebaseOptions;
import com.google.firebase.storage.FirebaseStorage;
import com.google.firebase.storage.StorageReference;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

import java.util.concurrent.TimeUnit;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class FlaskCommunication extends AppCompatActivity implements View.OnClickListener {
//    private static final String urls = "http://192.168.219.103:3333/comm";
    private static final int PICK_IMAGE_REQUEST = 1;
    private ApiController apiController;
    ImageButton btnDetect, btnChoicePickture;
    ImageView imgOrg, imgConv;
    Uri uri;

    File imageFile;

    @Override
    public void onClick(View view) {

        switch (view.getId()) {
            case R.id.btnDetect:
                if (imageFile==null) {
                    Toast.makeText(getApplicationContext(), "사진을 선택해주세요",Toast.LENGTH_SHORT).show();
                    return;
                }
                new Thread(()-> { apiController.getMaskFile(new MaskRequestDto(imageFile)); }).start();
                break;
            case R.id.btnChoicePickture:
                Intent intent = new Intent(Intent.ACTION_PICK);
                intent.setType("image/*");
                startActivityForResult(intent, 1);
                break;
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_flask_communication);

        apiController = new ApiControllerImpl();

        ActionBar actionBar = getSupportActionBar();
        actionBar.hide();

        btnDetect = findViewById(R.id.btnDetect);
        btnChoicePickture = findViewById(R.id.btnChoicePickture);
        imgOrg = findViewById(R.id.imgOrg);
        imgConv = findViewById(R.id.imgConv);
    }


    // 사진첩
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == PICK_IMAGE_REQUEST && resultCode == RESULT_OK && data != null) {
            Uri selectedImageUri = data.getData();

            // URI를 사용하여 파일 객체 생성
            imageFile = uriToFile(selectedImageUri);
        }
    }

    private File uriToFile(Uri selectedImageUri) {
        String[] filePathColumn = { MediaStore.Images.Media.DATA };
        Cursor cursor = getContentResolver().query(selectedImageUri, filePathColumn, null, null, null);
        if (cursor == null) return null;

        cursor.moveToFirst();

        int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
        String picturePath = cursor.getString(columnIndex);
        cursor.close();

        return new File(picturePath);
    }

    // 파이어베이스 사진 가져오기
//    private void getImageAndMask(String fileName){
//        FirebaseStorage storage = FirebaseStorage.getInstance();
//        StorageReference storageReference = storage.getReference();
//        StorageReference pathReference = storageReference.child("image/"+fileName);
//
//        if (pathReference == null) {
//            Toast.makeText(getApplicationContext(), "저장소에 사진이 없습니다." ,Toast.LENGTH_SHORT).show();
//        } else {
//            StorageReference submitProfile = storageReference.child("image/"+fileName);
//            submitProfile.getDownloadUrl().addOnSuccessListener(new OnSuccessListener<Uri>() {
//                @Override
//                public void onSuccess(Uri uri) {
//                    Glide.with(getApplicationContext()).load(uri).into(imgOrg);
//
//                }
//            }).addOnFailureListener(new OnFailureListener() {
//                @Override
//                public void onFailure(@NonNull Exception e) {
//
//                }
//            });
//
//            StorageReference submitProfileSec = storageReference.child("mask/"+fileName);
//            submitProfileSec.getDownloadUrl().addOnSuccessListener(new OnSuccessListener<Uri>() {
//                @Override
//                public void onSuccess(Uri uri) {
//                    Glide.with(getApplicationContext()).load(uri).into(imgConv);
//
//                }
//            }).addOnFailureListener(new OnFailureListener() {
//                @Override
//                public void onFailure(@NonNull Exception e) {
//
//                }
//            });
//        }
//    }

    // uri -> path 변환기
//    public String getRealPathFromURI(Uri contentUri) {
//        String[] proj = { MediaStore.Images.Media.DATA };
//
//        Cursor cursor = getContentResolver().query(contentUri, proj, null, null, null);
//        cursor.moveToNext();
//        String path = cursor.getString(cursor.getColumnIndex(MediaStore.MediaColumns.DATA));
//        Uri uri = Uri.fromFile(new File(path));
//
//        cursor.close();
//        return path;
//    }
//
//    // 플라스크 통신
//    public void sendServer(){
//        class sendData extends AsyncTask<Void, Void, String> {
//
//
//            @Override
//            protected void onPreExecute() {
//                super.onPreExecute();
//            }
//
//            @Override
//            protected void onPostExecute(String s) {
//                super.onPostExecute(s);
//            }
//
//            @Override
//            protected void onProgressUpdate(Void... values) {
//                super.onProgressUpdate(values);
//            }
//
//            @Override
//            protected void onCancelled(String s) {
//                super.onCancelled(s);
//            }
//
//            @Override
//            protected void onCancelled() {
//                super.onCancelled();
//            }
//
//            @Override
//            protected String doInBackground(Void... voids) {
//
//                 try {
//                    OkHttpClient client = new OkHttpClient.Builder()
//                            .connectTimeout(100, TimeUnit.SECONDS)
//                            .readTimeout(100,TimeUnit.SECONDS)
//                            .writeTimeout(100,TimeUnit.SECONDS)
//                            .build();
//
//                    RequestBody requestBody = new MultipartBody.Builder()
//                            .setType(MultipartBody.FORM)
//                            .addFormDataPart("title", "STORE Camera")
//                            .addFormDataPart("file", "file",RequestBody.create(MediaType.parse("image/jpg"), new File(getRealPathFromURI(uri))))
//                            .build();
//
//                    Request request = new Request.Builder()
//                            .post(requestBody)
//                            .url(urls)
//                            .build();
//
//                    Response responses = null;
//                    responses = client.newCall(request).execute();
//                    String fileName = responses.body().string();
//                    getImageAndMask(fileName);
//
//                    System.out.println(fileName);
//
//                }  catch (IOException e) {
//                    e.printStackTrace();
//                }
//                return null;
//            }
//        }
//        sendData sendData = new sendData();
//        sendData.execute();
//    }
}
