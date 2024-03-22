package com.example.carscratchdetector.controller;

import com.example.carscratchdetector.dto.MaskRequestDto;
import com.example.carscratchdetector.dto.MaskResponseDto;

import java.io.File;

import okhttp3.MultipartBody;
import retrofit2.Call;
import retrofit2.http.Headers;
import retrofit2.http.Multipart;
import retrofit2.http.POST;
import retrofit2.http.Part;

public interface MaskRetrofitController {
    @Multipart
    @POST("/api/v1/mask-file")
    Call<MaskResponseDto> getMaskFile(@Part MultipartBody.Part file);
}
