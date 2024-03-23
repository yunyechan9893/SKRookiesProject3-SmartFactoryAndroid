package com.example.carscratchdetector.view;

import android.content.Intent;


import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.appcompat.app.ActionBar;
import androidx.appcompat.app.AppCompatActivity;

import com.bumptech.glide.Glide;

import com.example.carscratchdetector.R;
import com.example.carscratchdetector.controller.ApiController;
import com.example.carscratchdetector.controller.impl.ApiControllerImpl;
import com.example.carscratchdetector.dto.MaskRequestDto;

import java.io.File;

public class FileUploadActivity extends AppCompatActivity implements View.OnClickListener {
    private static final int PICK_IMAGE_REQUEST = 1;
    private ApiController apiController;
    ImageButton btnDetect, btnChoicePickture;
    ImageView imgOrg, imgConv;
    File imageFile;

    @Override
    public void onClick(View view) {

        switch (view.getId()) {
            case R.id.btnDetect:
                if (imageFile==null) {
                    Toast.makeText(getApplicationContext(), "사진을 선택해주세요",Toast.LENGTH_SHORT).show();
                    return;
                }

                apiController.getMaskFile(new MaskRequestDto(imageFile), maskImg -> {
                    // 성공 콜백: UI 스레드에서 ImageView에 Bitmap 설정
                    runOnUiThread(() -> {
                        imgConv.setImageBitmap(maskImg);
                    });
                }, error -> {
                    // 실패 콜백: 오류 처리
                    Log.e("API_CALL", "Error during API call", error);
                });


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
            Glide.with(getApplicationContext()).load(selectedImageUri).into(imgOrg);

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
}
