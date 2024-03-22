package com.example.carscratchdetector.dto;

import com.google.gson.annotations.SerializedName;

public class MaskResponseDto {
    @SerializedName("content")
    String maskImg;

    @SerializedName("message")
    String message;
}
