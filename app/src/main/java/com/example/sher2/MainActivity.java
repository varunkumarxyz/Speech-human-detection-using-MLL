package com.example.sher2;
import android.Manifest;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.channels.WritableByteChannel;
import java.util.Arrays;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";
    private static final int REQUEST_RECORD_AUDIO_PERMISSION = 200;
    private static final int SAMPLE_RATE = 16000;
    private static final int BUFFER_SIZE = SAMPLE_RATE * 2;
    private static final String MODEL_FILE_NAME = "model.tflite";
    private static final String[] EMOTIONS = {"angry", "disgust", "fear", "happy", "neutral", "sad", "surprise", "calm"};

    private Button startButton;
    private TextView emotionTextView;
    private ImageView emotionImageView;

    private boolean isRecording = false;
    private AudioRecord audioRecord;
    private Thread recordingThread;
    private ByteBuffer audioBuffer;
    private File outputFile;
    private Interpreter tflite;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        startButton = findViewById(R.id.startButton);
        emotionTextView = findViewById(R.id.emotionTextView);
        emotionImageView = findViewById(R.id.emotionImageView);

        // Check if the app has permission to record audio
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[] { Manifest.permission.RECORD_AUDIO }, REQUEST_RECORD_AUDIO_PERMISSION);
        }
        // Load the TensorFlow Lite model
        try {
            tflite = new Interpreter(loadModelFile());
        } catch (IOException e) {
            Log.e(TAG, "Failed to load TensorFlow Lite model.", e);
        }

        startButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (!isRecording) {
                    startRecording();
                    startButton.setText("Stop Recording");
                } else {
                    stopRecording();
                    startButton.setText("Start Recording");
                }
            }
        });
    }

    // Request permission to record audio
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_RECORD_AUDIO_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "Permission granted to record audio.", Toast.LENGTH_SHORT).show();
            } else {
                Toast.makeText(this, "Permission to record audio denied.", Toast.LENGTH_SHORT).show();
            }
        }
    }

    // Load the TensorFlow Lite model from the assets folder
    private ByteBuffer loadModelFile() throws IOException {
        try (FileInputStream inputStream = new FileInputStream(new File(getCacheDir(), MODEL_FILE_NAME))) {
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileChannel.position();
            long declaredLength = fileChannel.size() - startOffset;
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        }
    }

    // Start recording audio from the microphone
    private void startRecording() {
        audioBuffer = ByteBuffer.allocateDirect(BUFFER_SIZE).order(ByteOrder.nativeOrder());
        audioRecord = new AudioRecord(MediaRecorder.AudioSource.DEFAULT, SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, BUFFER_SIZE);
        audioRecord.startRecording();
        isRecording = true;
        recordingThread = new Thread(new Runnable() {
            @Override
            public void run() {
                writeAudioDataToFile();
                try {
                    processAudio();
                } catch (IOException e) {
                    Log.e(TAG, "Failed to process audio.", e);
                }
            }
        });
        recordingThread.start();
    }

    // Stop recording audio from the microphone
    private void stopRecording() {
        isRecording = false;
        audioRecord.stop();
        recordingThread = null;
    }

    // Write audio data to file as a .wav file
    private void writeAudioDataToFile() {
        outputFile = new File(getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS), "recording.wav");
        try (FileOutputStream fos = new FileOutputStream(outputFile)) {
            WritableByteChannel channel = Channels.newChannel(fos);
            while (isRecording) {
                int bytesRead = audioRecord.read(audioBuffer, BUFFER_SIZE);
                if (bytesRead > 0) {
                    audioBuffer.limit(bytesRead);
                    channel.write(audioBuffer);
                    audioBuffer.clear();
                }
            }
        } catch (FileNotFoundException e) {
            Log.e(TAG, "Failed to open file output stream.", e);
        } catch (IOException e) {
            Log.e(TAG, "Failed to write audio data to file.", e);
        }
    }

    // Process the recorded audio and get the predicted emotion from the TensorFlow Lite model
    private void processAudio() throws IOException {
        // Convert the .wav file to a 16-bit, 16kHz mono PCM array
        short[] audioData
        try (AudioInputStream ais = AudioSystem.getAudioInputStream(outputFile)) {
            AudioFormat format = ais.getFormat();
            int numBytes = ais.available();
            int numSamples = numBytes / (format.getChannels() * format.getSampleSizeInBits() / 8);
            audioData = new short[numSamples];
            int bytesRead = ais.read(audioData);
            if (bytesRead != numSamples) {
                Log.e(TAG, "Failed to read expected number of samples from .wav file.");
                return;
            }
        } catch (UnsupportedAudioFileException | IOException e) {
            Log.e(TAG, "Failed to read .wav file.", e);
            return;
        }

        // Normalize the audio data to [-1.0, 1.0]
        float[] floatAudioData = new float[audioData.length];
        for (int i = 0; i < audioData.length; i++) {
            floatAudioData[i] = audioData[i] / 32767.0f;
        }

        // Preprocess the audio data
        float[][][] inputBuffer = new float[1][SPEC_HEIGHT][SPEC_WIDTH];
        float[] spectrogram = AudioPreprocessing.calculateLogSpectrogram(floatAudioData, SAMPLE_RATE);
        for (int i = 0; i < SPEC_HEIGHT; i++) {
            for (int j = 0; j < SPEC_WIDTH; j++) {
                inputBuffer[0][i][j] = spectrogram[i * SPEC_WIDTH + j];
            }
        }

        // Run inference on the model
        tflite.run(inputBuffer, outputBuffer);

        // Get the predicted emotion from the output buffer
        int maxIndex = 0;
        float maxValue = outputBuffer[0][0];
        for (int i = 1; i < NUM_OUTPUTS; i++) {
            if (outputBuffer[0][i] > maxValue) {
                maxIndex = i;
                maxValue = outputBuffer[0][i];
            }
        }

        // Show the predicted emotion and corresponding image
        String emotion = EMOTIONS[maxIndex];
        int drawableId = getResources().getIdentifier(emotion, "drawable", getPackageName());
        Drawable drawable = getResources().getDrawable(drawableId, null);
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                resultTextView.setText("You seem to be " + emotion + ".");
                imageView.setImageDrawable(drawable);
            }
        });
    }




