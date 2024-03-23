package com.example.carscratchdetector.controller;

import android.graphics.Bitmap;

import com.example.carscratchdetector.dto.MaskRequestDto;
import com.example.carscratchdetector.dto.MaskResponseDto;

import java.io.File;
import java.util.function.Consumer;

import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.POST;

public interface ApiController {
    void getMaskFile(MaskRequestDto maskRequestDto, Consumer<Bitmap> onSuccess, Consumer<Throwable> onFailure);
}
