/**
 * 
 */
package org.MachineLearing.BP;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;



/**
 * @author Mor
 *
 */
public class BPTest {

	/**
	 * @param args
	 */
	
	private static double Valuemin = 10;
	private static double Valuemax = -10;
	private static final int triantimes = 500;
	
	public static void main(String[] args) throws IOException{
		
		BPAlog Bp = new BPAlog(4, 5, 3);

		List<Double> trainDatalist = new ArrayList<Double>();
		List<Double> trainTargetlist = new ArrayList<Double>();
		List<Double> realDatalist = new ArrayList<Double>();
		List<Double> realTargetlist = new ArrayList<Double>();

		double[] TrainDataMax = new double[4];
		double[] TrainDataMin = new double[4];
		double[] realDataMax = new double[4];
		double[] realDataMin = new double[4];

		String TrainfilePath = "/Users/Mor/Documents/MachineLearningData/BpData-iris/train.txt";
		String realfilePath = "/Users/Mor/Documents/MachineLearningData/BpData-iris/real.txt";

		ReadData(TrainfilePath, TrainDataMax, TrainDataMin, trainDatalist, trainTargetlist);
		ReadData(realfilePath, realDataMax, realDataMin, realDatalist, realTargetlist);

		Map<Integer, List<Double>> TrainData = PreData(trainDatalist, trainTargetlist);
		Map<Integer, List<Double>> MRealData = PreData(trainDatalist, trainTargetlist);

		for (int i = 0; i < triantimes; i++) {
			for (int j = 0; j < TrainData.size(); j++) {

				List<Double> list = TrainData.get(j);
				double trainData[] = new double[4];
				double targetData[] = new double[3];

				for (int k = 0; k < list.size() - 1; k++)
					trainData[k] = normalization(list.get(k), TrainDataMax[k], TrainDataMin[k]);
				if (list.get(4) == 0)
					targetData[0] = 1;
				else if (list.get(4) == 1)
					targetData[1] = 1;
				else
					targetData[2] = 1;

				Bp.train(trainData, targetData);
			}

		}
		System.out.println("Train Over");

		for (int i = 0; i < MRealData.size(); i++) {
			List<Double> list = MRealData.get(i);
			double RealData[] = new double[4];
			double Realtarget = 0d;

			for (int k = 0; k < list.size() - 1; k++)
				RealData[k] = normalization(list.get(k), realDataMax[k], realDataMin[k]);
			Realtarget = list.get(4);

			double[] relust = Bp.test(RealData);

			double max = -Integer.MIN_VALUE;
			int idx = -1;

			for (int j = 0; j < relust.length; j++) {
				if (relust[j] > max) {
					max = relust[j];
					idx = j;
				}
			}
			switch (idx) {
			case 0:
				System.out.println("Iris-setosa，0-" + (int) Realtarget);
				break;
			case 1:
				System.out.println("Iris-versicolor，1-" + (int) Realtarget);
				break;
			case 2:
				System.out.println("Iris-virginica，2-" + (int) Realtarget);
				break;

			}

		}

		System.out.println(Bp.hiddenErrSum);
		System.out.println(Bp.outputErrSum);

	}



/**
 * @param trainDatalist
 * @param trainTargetlist
 * @return data set which a group of five
 */
public static Map<Integer, List<Double>> PreData(List<Double> trainDatalist, List<Double> trainTargetlist) {
	Map<Integer, List<Double>> map = new HashMap<Integer, List<Double>>();
	int k = 0;
	for (int i = 0; i < trainDatalist.size(); i += 4) {
		List<Double> list = new ArrayList<Double>();
		for (int j = i; j < i + 4; j++) {
			list.add(trainDatalist.get(j));
		}
		list.add(trainTargetlist.get(k));
		map.put(k, list);
		k++;
	}
	return map;
}

/**
 * @param filePath
 * @param ValueMax
 * @param ValueMin
 * @param list
 * @param reallist
 * 
 */
public static void ReadData(String filePath, double[] ValueMax, double[] ValueMin, List<Double> list,
		List<Double> reallist) {

	try {
		String encoding = "GBK";
		File file = new File(filePath);

		if (file.isFile() && file.exists()) {

			InputStreamReader read = new InputStreamReader(new FileInputStream(file), encoding);

			BufferedReader bufferedreader = new BufferedReader(read);
			String linetxt = null;

			while ((linetxt = bufferedreader.readLine()) != null) {
				String[] a = linetxt.split(",");
				for (int i = 0; i < a.length - 1; i++) {
					if (a[i] != null && !a[i].trim().equals("")) {
						try {
							double value = Double.parseDouble(a[i]);
							list.add(value);
							ValueMax[i] = getMax(value);
							ValueMin[i] = getMin(value);
						} catch (Exception e) {
							System.out.println("error Message：" + e.getMessage());
						}
					}
				} // System.out.println(a.length);
				if (a[4] != null && !a[4].trim().equals(" ")) {
					try {
						if (a[4] != null && a[4].trim().equals("Iris-setosa"))
							reallist.add(0d);
						else if (a[4] != null && a[4].trim().equals("Iris-versicolor"))
							reallist.add(1d);
						else
							reallist.add(2d);
					} catch (Exception e) {
						System.out.println("error Message：" + e.getMessage());
					}
				}
			}
			read.close();

		} else
			System.out.println("no file!");

	} catch (Exception e) {
		System.out.println("file is worng");
		e.printStackTrace();
	}

}

/**
 * @param x
 * @param max
 * @param min
 * @return normalization factor
 */
public static double normalization(double x, double max, double min) {
	double value = 0d;
	value = (x - min) / (max - min);
	return value;
}

/**
 * @param val
 * @return Valuemax
 */

public static double getMax(double val) {
	// double max =val;
	if (Valuemax < val)
		Valuemax = val;
	return Valuemax;
}

/**
 * @param val
 * @return Valuemin
 */

public static double getMin(double val) {
	// double min = 10;
	if (Valuemin > val)Valuemin = val;
	return Valuemin;
	}
}
