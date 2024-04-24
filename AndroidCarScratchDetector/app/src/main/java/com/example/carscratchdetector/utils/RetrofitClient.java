package com.example.carscratchdetector.utils;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.util.concurrent.TimeUnit;

import okhttp3.OkHttpClient;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class RetrofitClient {
    private static Retrofit instance = null;
    // 테스트 시
    // private static final String BASE_URL =  "http://10.0.2.2:3333/";
    private static final String BASE_URL = "http://api.smartfactory.yechan-portfolio.site";
    private static Gson gson = new GsonBuilder().setLenient().create();

    private RetrofitClient() {
    }

    public static Retrofit getInstance() {
        if (instance == null) {

            OkHttpClient.Builder httpClient = new OkHttpClient.Builder()
                    .connectTimeout(30, TimeUnit.SECONDS)  // 연결 타임아웃 설정
                    .readTimeout(30, TimeUnit.SECONDS)     // 읽기 타임아웃 설정
                    .writeTimeout(30, TimeUnit.SECONDS);   // 쓰기 타임아웃 설정


            instance = new Retrofit.Builder()
                    .baseUrl(BASE_URL)
                    .addConverterFactory(GsonConverterFactory.create(gson))
                    .client(httpClient.build())
                    .build();
        }
        return instance;
    }

}