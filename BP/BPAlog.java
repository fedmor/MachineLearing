/**
 * 
 */
package org.MachineLearing.BP;

import java.util.Random;

/**
 * @author Mor
 *
 */
public class BPAlog {

	private final double[] input;
	private final double[] hidden;
	private final double[] output;
	private final double[] target;
	private final double[] hinDelta;
	private final double[] optDelta;

	private final double[][] inhinWeight;
	private final double[][] hinoptWeight;

	private final double eta; // learning rate
	private final double momentum;// move momentum

	private final double[][] inhinPerWeight;
	private final double[][] hinoptPerWeight;

	private final Random random;

	public double hiddenErrSum = 0d;
	public double outputErrSum = 0d;

	public BPAlog(int inputSize, int hiddenSize, int optSize, double eta, double momentum) { // 网络初始化

		input = new double[inputSize + 1];
		hidden = new double[hiddenSize + 1];
		output = new double[optSize + 1];
		target = new double[optSize + 1];

		inhinWeight = new double[inputSize + 1][hiddenSize + 1];
		hinoptWeight = new double[hiddenSize + 1][optSize + 1];
		inhinPerWeight = new double[inputSize + 1][hiddenSize + 1];
		hinoptPerWeight = new double[hiddenSize + 1][optSize + 1];

		hinDelta = new double[hiddenSize + 1];
		optDelta = new double[optSize + 1];

		this.eta = eta;
		this.momentum = momentum;

		random = new Random(19881211);

		randomWeights(inhinWeight);
		randomWeights(hinoptWeight);

	}

	public void randomWeights(double[][] matrix) { // 网络权值初始化

		for (int i = 0, len_i = matrix.length; i < len_i; i++) {
			for (int j = 0, len_j = matrix[i].length; j < len_j; j++) {
				double real = random.nextDouble();
				matrix[i][j] = random.nextDouble() > 0.5 ? real : -real;
			}
		}

	}

	public BPAlog(int inputSize, int hiddenSize, int outputSize) { // 方法重写
		this(inputSize, hiddenSize, outputSize, 0.25, 0.9);

	}

	/*
	 * 网络计算代码块 /
	 */
	public double[] test(double[] inData) {
		loadData(inData, input);
		forward();
		return getNetworkOpt();
	}

	public double[] getNetworkOpt() {
		int len = output.length;
		double[] temp = new double[len - 1];
		for (int i = 1; i < len; i++) {
			temp[i - 1] = output[i];
		}
		return temp;
	}
	/*
	 * 网络训练代码块 /
	 */

	public void train(double[] trainData, double[] trainTarget) { // 网络训练函数
		loadData(trainData, input); // 样本导入
		loadData(trainTarget, target); // 样本结果导入
		forward(); // 正向传导
		/*
		 * 网络到反向传导 误差计算以及权值调整
		 */
		calculateDelta();
		adjustWeight();
	}

	/*
	 * 权值调整代码块 /
	 */
	private void adjustWeight() {
		adjustWeight(hinDelta, input, inhinWeight, inhinPerWeight);
		adjustWeight(optDelta, hidden, hinoptWeight, hinoptPerWeight);
	}

	private void adjustWeight(double[] Delta, double[] layer, double[][] Weight, double[][] PervWeight) {
		layer[0] = 1;
		for (int i = 1, len_i = Delta.length; i < len_i; i++) {
			for (int j = 0, len_j = layer.length; j < len_j; j++) {
				double newval = momentum * PervWeight[j][i] + eta * Delta[i] * layer[j];
				Weight[j][i] += newval;
				PervWeight[j][i] = newval;
			}
		}
	}

	/*
	 * 正向传递代码块 /
	 */
	private void forward() {
		forward(input, hidden, inhinWeight);
		forward(hidden, output, hinoptWeight);
	}

	private void forward(double[] in, double[] out, double[][] Weight) {

		in[0] = 1d;
		for (int i = 1, len_out = out.length; i < len_out; i++) {
			double sum = 0;
			for (int j = 0, len_in = in.length; j < len_in; j++)
				sum += in[j] * Weight[j][i];
			out[i] = sigmoid(sum);
		}
	}

	/*
	 * 误差计算代码块 /
	 */
	private void calculateDelta() {

		calculateouputErr();
		calculatehinddenErr();

	}

	private void calculatehinddenErr() {// 隐层误差计算

		double sumErr = 0;
		for (int i = 1, len_i = hinDelta.length; i < len_i; i++) {
			double hid = hidden[i];
			double sum = 0;
			for (int j = 1, len_j = optDelta.length; j < len_j; j++)
				sum += hinoptWeight[i][j] * optDelta[j];
			hinDelta[i] = hid * (1d - hid) * sum;
			sumErr += Math.abs(hinDelta[i]);
		}
		hiddenErrSum = sumErr;
	}

	private void calculateouputErr() { // 输出层误差计算

		double sumErr = 0;
		for (int i = 1, len = optDelta.length; i < len; i++) {
			double opt = output[i];
			optDelta[i] = opt * (1d - opt) * (target[i] - opt);
			sumErr = sumErr + Math.abs(optDelta[i]);
		}
		outputErrSum = sumErr;
	}

	/*
	 * 数据导入函数 /
	 */
	private void loadData(double[] toData, double[] targetArray) {
		for (int i = 1, len = toData.length; i <= len; i++) {
			targetArray[i] = toData[i - 1];
		}
	}

	private double sigmoid(double val) { // 激活函数
		return 1d / (1d + Math.exp(-val));
	}

}
