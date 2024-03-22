package com.example.carscratchdetector.controller;

import com.example.carscratchdetector.dto.MaskRequestDto;
import com.example.carscratchdetector.dto.MaskResponseDto;

import java.io.File;

import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.POST;

public interface ApiController {
    Call<MaskResponseDto> getMaskFile(@Body MaskRequestDto maskRequestDto);
}
