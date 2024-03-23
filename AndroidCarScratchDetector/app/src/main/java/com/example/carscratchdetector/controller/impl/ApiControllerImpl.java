package com.example.carscratchdetector.controller.impl;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Build;
import android.util.Base64;
import android.util.Log;

import com.example.carscratchdetector.controller.ApiController;
import com.example.carscratchdetector.controller.MaskRetrofitController;
import com.example.carscratchdetector.dto.MaskRequestDto;
import com.example.carscratchdetector.dto.MaskResponseDto;
import com.example.carscratchdetector.utils.RetrofitClient;

import java.util.function.Consumer;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class ApiControllerImpl implements ApiController {

    @Override
    public void getMaskFile(MaskRequestDto maskRequestDto, Consumer<Bitmap> onSuccess, Consumer<Throwable> onFailure) {
        RequestBody requestFile = RequestBody.create(MediaType.parse("image/jpeg"), maskRequestDto.image);
        MultipartBody.Part body = MultipartBody.Part.createFormData("file", maskRequestDto.image.getName(), requestFile);

        MaskRetrofitController service = RetrofitClient.getInstance().create(MaskRetrofitController.class);
        Call<MaskResponseDto> call = service.getMaskFile(body);

        call.enqueue(new Callback<MaskResponseDto>() {
            @Override
            public void onResponse(Call<MaskResponseDto> call, Response<MaskResponseDto> response) {
                if (response.isSuccessful() && response.body() != null) {
                    Bitmap bitmap = decodeBase64ToBitmap(response.body().getMaskImg());
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                        onSuccess.accept(bitmap);
                    }
                } else {
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                        onFailure.accept(new RuntimeException("Response is not successful"));
                    }
                }
            }

            @Override
            public void onFailure(Call<MaskResponseDto> call, Throwable t) {
                onFailure.accept(t);
            }
        });
    }

    private Bitmap decodeBase64ToBitmap(String base64Str) {
        // Base64 문자열의 데이터 부분만 디코딩합니다.
        byte[] decodedBytes = Base64.decode(base64Str, Base64.DEFAULT);

        // 디코딩된 바이트 배열을 사용하여 Bitmap을 생성합니다.
        return BitmapFactory.decodeByteArray(decodedBytes, 0, decodedBytes.length);
    }
}
