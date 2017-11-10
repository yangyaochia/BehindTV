import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 * Normal.getMap -> get attribute name by matrix col index
 * Normal.getGenMap -> get matrix col index by attribute name
 * Normal.getMat -> get matrix
 * Normal.getRating -> get rating
 * Normal.getImdbRating -> get imdb rating
 * Normal.getTitles -> get all titles
 */
public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        String json = "";
        ArrayList<Show> shows = null;
        Normal norm = null;
        Gson gson = new Gson();
        NelderMead findFeatureWeighting = null;
        //SVD

        json = Tool.readFile("show.json");
        shows = gson.fromJson(json, new TypeToken<ArrayList<Show>>() {
        }.getType());
        norm = new Normal(shows);

        norm.normFeatures();
        //norm.printMat();// test

        HashMap<Integer, String> hmap = norm.getMap();
        // Display content using Iterator
        Set set = hmap.entrySet();
        Iterator iterator = set.iterator();
        try {
            //All your IO Operations
            FileWriter fw = new FileWriter("Feature.csv");
            while (iterator.hasNext()) {
                Map.Entry mentry = (Map.Entry) iterator.next();
                fw.append(String.valueOf(mentry.getValue()) + "\n");
            }
            fw.append("\n");
            fw.close();
        } catch (IOException ioe) {
            //Handle exception here, most of the time you will just log it.
        }
        ArrayList<String> titles = norm.getTitles();
        try {
            //All your IO Operations
            FileWriter fw = new FileWriter("Title.csv");
            for (int i = 0; i < titles.size(); i++)
                fw.append(String.valueOf(i + 1) + ":" + String.valueOf(titles.get(i)) + "\n");
            fw.append("\n");
            fw.close();
        } catch (IOException ioe) {
            //Handle exception here, most of the time you will just log it.
        }
        int kFold = 5;  // k-fold cross-validation
        long time1, time2, time3;


        time1 = System.currentTimeMillis();

        // doSomething()

        double[][] mat = norm.getMat();
        //double[][] mat = svd.getU();
        double[][] matA = new double[mat.length][39];
        double[][] matAB = new double[mat.length][45];
        //double[][] matAC = new double[mat.length][mat[0].length-6];
        //double[][] matB = new double[mat.length][6];
        //double[][] matC = new double[mat.length][mat[0].length-45];
        //System.out.println("mat.length-45 = "+mat.length);

        for ( int i = 0 ; i < matA.length ; i++ ) {
            for ( int j = 0 ; j < matA[i].length ; j++ ) {
                matA[i][j] = mat[i][j];
            }
        }
        for ( int i = 0 ; i < matAB.length ; i++ ) {
            for ( int j = 0 ; j < matAB[i].length ; j++ ) {
                matAB[i][j] = mat[i][j];
            }
        }
/*        for ( int i = 0 ; i < matAC.length ; i++ ) {
            for ( int j = 0 ; j < matA[i].length ; j++ ) {
                matAC[i][j] = mat[i][j];
            }
        }
        for ( int i = 0 ; i < matAC.length ; i++ ) {
            for ( int j = matA.length ; j < matAC[i].length ; j++ ) {
                matAC[i][j] = mat[i][j+6];
            }
        }

        for ( int i = 0 ; i < matB.length ; i++ ) {
            for ( int j = 0 ; j < matB[i].length ; j++ ) {
                matB[i][j] = mat[i][j+39];
            }
        }
        for ( int i = 0 ; i < matC.length ; i++ ) {
            for ( int j = 0 ; j < matC[i].length ; j++ ) {
                //System.out.print("j = "+j);
                matC[i][j] = mat[i][j+45];
            }
        }*/

        SVD s4 = new SVD(matA);
        s4.buildSVD();
        SVD s5 = new SVD(matAB);
        s5.buildSVD();
        SVD s6 = new SVD(mat);
        s6.buildSVD();
/*        SVD s7 = new SVD(matAC);
        s7.buildSVD();

        SVD s8 = new SVD(matB);
        s7.buildSVD();
        SVD s9 = new SVD(matC);
        s7.buildSVD();*/

        ArrayList<Double> rating = norm.getRating();
        ArrayList<Double> results = new ArrayList<>();
        double[] ratingArray = new double[mat.length];

        double result = 0;

        Prediction p1 = new Prediction(matA, norm.getImdbRating(), kFold);
        p1.predict();
        result = p1.getMeanSquaredError();
        results.add(result);

        Prediction p2 = new Prediction(matAB, norm.getImdbRating(), kFold);
        p2.predict();
        result = p2.getMeanSquaredError();
        results.add(result);

        Prediction p3 = new Prediction(mat, norm.getImdbRating(), kFold);
        p3.predict();
        result = p3.getMeanSquaredError();
        results.add(result);

        Prediction p4 = new Prediction(s4.getU(), norm.getImdbRating(), kFold);
        p4.predict();
        result = p4.getMeanSquaredError();
        results.add(result);

        Prediction p5 = new Prediction(s5.getU(), norm.getImdbRating(), kFold);
        p5.predict();
        result = p5.getMeanSquaredError();
        results.add(result);

        Prediction p6 = new Prediction(s6.getU(), norm.getImdbRating(), kFold);
        p6.predict();
        result = p6.getMeanSquaredError();
        results.add(result);

        /*Prediction p7 = new Prediction(s7.getU(), norm.getImdbRating(), kFold);
        p7.predict();
        result = p7.getMeanSquaredError();
        results.add(result);*/
/*        Prediction p8 = new Prediction(matB, norm.getImdbRating(), kFold);
        p8.predict();
        result = p8.getMeanSquaredError();
        results.add(result);

        Prediction p9 = new Prediction(matC, norm.getImdbRating(), kFold);
        p9.predict();
        result = p9.getMeanSquaredError();
        results.add(result);

        Prediction p10 = new Prediction(s9.getU(), norm.getImdbRating(), kFold);
        p10.predict();
        result = p10.getMeanSquaredError();
        results.add(result);*/

        System.out.println("Training finished! Press Enter to continue...");
        scanner.nextLine();

        Tool.printResult(results);
    }
}
