package com.example.carscratchdetector.controller.impl;

import android.util.Log;

import com.example.carscratchdetector.controller.ApiController;
import com.example.carscratchdetector.controller.MaskRetrofitController;
import com.example.carscratchdetector.dto.MaskRequestDto;
import com.example.carscratchdetector.dto.MaskResponseDto;
import com.example.carscratchdetector.utils.RetrofitClient;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class ApiControllerImpl implements ApiController {

    @Override
    public Call<MaskResponseDto> getMaskFile(MaskRequestDto maskRequestDto) {
        RequestBody requestFile = RequestBody.create(MediaType.parse("image/jpeg"), maskRequestDto.image);
        MultipartBody.Part body = MultipartBody.Part.createFormData("file", maskRequestDto.image.getName(), requestFile);

        MaskRetrofitController service = RetrofitClient.getInstance().create(MaskRetrofitController.class);
        Call<MaskResponseDto> call = service.getMaskFile(body);
        call.enqueue(new Callback<MaskResponseDto>() {
            @Override
            public void onResponse(Call<MaskResponseDto> call, Response<MaskResponseDto> response) {
                Log.d("call:",response.toString());
            }

            @Override
            public void onFailure(Call<MaskResponseDto> call, Throwable t) {
                Log.d("call:",t.toString());
            }
        });

        return null;
    }
}
