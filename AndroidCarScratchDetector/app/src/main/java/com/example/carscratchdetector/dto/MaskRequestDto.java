package com.example.carscratchdetector.dto;

import java.io.File;

public class MaskRequestDto {
    public File image;

    public MaskRequestDto(){

    }

    public MaskRequestDto(File imageFile) {
        this.image = imageFile;
    }
}
