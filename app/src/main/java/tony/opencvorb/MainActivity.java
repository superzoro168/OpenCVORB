package tony.opencvorb;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Environment;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.ImageView;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";
    private static final String mobile_path = Environment
            .getExternalStorageDirectory().toString() + "/OpenCVORB_folder/";

    private ImageView imageView;
    private TextView matchPointTv;
    private Bitmap imageMatched;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imageView = (ImageView) findViewById(R.id.imageView);
        matchPointTv = (TextView) findViewById(R.id.matchPointTV);
    }

    private BaseLoaderCallback mOpenCVCallBack = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");

                    /*
                     * 進行影像處理
                     */

                    Mat img1 = Imgcodecs.imread(mobile_path+"/p01.jpg", 0);
                    Mat img2 = Imgcodecs.imread(mobile_path+"/p02.jpg", 0);

                    int match_points = detect(img1, img2);

                    matchPointTv.setText("match point:"+match_points);

                    imageView.setImageBitmap(imageMatched);

                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    //此方法為偵測兩張圖的特徵點
    public Integer detect(Mat img1, Mat img2) {
        Size sz = new Size(300, 300);

        FeatureDetector detector = FeatureDetector.create(FeatureDetector.ORB);
        DescriptorExtractor descriptor = DescriptorExtractor
                .create(DescriptorExtractor.ORB);

        DescriptorMatcher matcher = DescriptorMatcher
                .create(DescriptorMatcher.BRUTEFORCE_HAMMING);

        Mat resizeimage = new Mat();
        Imgproc.resize(img1, resizeimage, sz);
        img1 = resizeimage;

        Mat descriptors1 = new Mat();
        MatOfKeyPoint keypoints1 = new MatOfKeyPoint();
        detector.detect(img1, keypoints1);
        descriptor.compute(img1, keypoints1, descriptors1);

        Mat resizeimage2 = new Mat();
        Imgproc.resize(img2, resizeimage2, sz);
        img2 = resizeimage2;

        Mat descriptors2 = new Mat();
        MatOfKeyPoint keypoints2 = new MatOfKeyPoint();
        detector.detect(img2, keypoints2);
        descriptor.compute(img2, keypoints2, descriptors2);

        MatOfDMatch matches = new MatOfDMatch();
        matcher.match(descriptors1, descriptors2, matches);
        int DIST_LIMIT = 80;
        List<DMatch> matchesList = matches.toList();
        List<DMatch> matches_final = new ArrayList<DMatch>();

        for (int i = 0; i < matchesList.size(); i++)
            if (matchesList.get(i).distance <= DIST_LIMIT) {
                matches_final.add(matches.toList().get(i));
            }// end if

        MatOfDMatch matches_final_mat = new MatOfDMatch();
        matches_final_mat.fromList(matches_final);

        int mTotalMatches = matches_final.size();

        Scalar RED = new Scalar(255, 0, 0);
        Scalar GREEN = new Scalar(0, 255, 0);

        Mat outputImg = new Mat();
        MatOfByte drawnMatches = new MatOfByte();

        Features2d.drawMatches(img1, keypoints1, img2, keypoints2,
                matches_final_mat, outputImg, GREEN, RED, drawnMatches,
                Features2d.NOT_DRAW_SINGLE_POINTS);

        imageMatched = Bitmap.createBitmap(outputImg.cols(),
                outputImg.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(outputImg, imageMatched);

        return mTotalMatches;
    }// end detect

    /** Call on every application resume **/
    @Override
    protected void onResume() {
        Log.i(TAG, "Called onResume");
        super.onResume();

        Log.i(TAG, "Trying to load OpenCV library");
        if (!OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mOpenCVCallBack)) {
            Log.e(TAG, "Cannot connect to OpenCV Manager");
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }
}
