#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
//#include "Eigen\core"
#include "narivectorpp.h"
#include "info.h"
#include <Eigen/Dense>
#include <sys/stat.h>
#include "direct.h"
#include <naricommon.h>
#include <nariinfocontroller.h>
#include <narifile.h>

template< class T >
void write_vector(std::vector<T> &v, const std::string filename) {
	FILE *fp;
	if (fopen_s(&fp, filename.c_str(), "wb") != 0) {
		std::cerr << "Cannot open file: " << filename << std::endl;
		std::abort();
	}
	fwrite(v.data(), sizeof(T), v.size(), fp);
	fclose(fp);
}

long get_file_size(std::string filename)
{
	FILE *fp;
	struct stat st;
	if (fopen_s(&fp, filename.c_str(), "rb") != 0) {
		std::cerr << "Cannot open file: " << filename << std::endl;
		std::abort();
	}
	fstat(_fileno(fp), &st);
	fclose(fp);
	return st.st_size;
}

template< class T >
void read_vector(std::vector<T> &v, const std::string filename) {

	auto num = get_file_size(filename) / sizeof(T);
	FILE *fp;
	if (fopen_s(&fp, filename.c_str(), "rb") != 0) {
		std::cerr << "Cannot open file: " << filename << std::endl;
		std::abort();
	}
	v.resize(num);
	fread(v.data(), sizeof(T), num, fp);
	fclose(fp);
}

template<typename T>
void write_matrix_raw_and_txt(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& data, std::string filename)
{
	//////////////////////////////////////////////////////////////
	// Wの書き出し												//
	// rowが隠れ層の数，colが可視層の数							//			
	// 重みの可視化を行う場合は，各行を切り出してreshapeを行う  //
	//////////////////////////////////////////////////////////////
	size_t rows = data.rows();
	size_t cols = data.cols();
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Data;
	Data = data;
	std::ofstream fs1(filename + ".txt");
	fs1 << "rows = " << rows << std::endl;
	fs1 << "cols = " << cols << std::endl;
	fs1 << typeid(Data).name() << std::endl;
	fs1.close();
	std::vector<T> save_data(rows * cols);
	Data.resize(rows * cols, 1);
	for (size_t i = 0; i < save_data.size(); i++)
		save_data[i] = Data(i, 0);
	write_vector(save_data, filename + ".raw");
	Data.resize(rows, cols);
}


void main(int argc, char *argv[]) {
	info input_info;
	input_info.input(argv[1]);

	//テキストデータ読み込み	
	std::vector<std::string> fcase;
	std::vector<std::string> rcase;
	std::ifstream f_case(input_info.dir_list + input_info.case_flist);
	std::ifstream r_case(input_info.dir_list + input_info.case_rlist);
	std::string buf_ft;
	std::string buf_rt;
	while (f_case&& getline(f_case, buf_ft))
	{
		fcase.push_back(buf_ft);
	}
	while (r_case&& getline(r_case, buf_rt))
	{
		rcase.push_back(buf_rt);
	}
	//学習全データ数,それぞれの次元数
	int n = fcase.size() - 1;
	int Fd = input_info.fd;
	int Rd = input_info.rd;
	double b = input_info.B;

	for (int i = 0; i < fcase.size(); i++) {
		//ファイル読み込み
		//成長前形状LS主成分スコア
		std::vector<double> Fl;
		read_vector(Fl, input_info.dir_score + "Fl/" + fcase[i] + "/mat.raw");
		// 成長後形状LS主成分スコア
		std::vector<double> Ref;
		read_vector(Ref, input_info.dir_score + "Ref/" + rcase[i] + "/mat.raw");
		//それぞれ学習,テストデータのスコアのみ抜き出す
		std::vector<double> Fl_tr;
		std::vector<double> Ref_tr;   //テストの入力
		std::vector<double> Fl_te;
		std::vector<double> Ref_te;   //テスト正解出力
		for (int j = 0; j < fcase.size(); j++) {
			for (int k = 0; k < Fd; k++) {
				//デバッグの時はここを変更するべし
				//int s = j*Fd + k;
				int s = j*(fcase.size() - 2) + k;
				if ((j == i) && (k < Fd)) {
					Fl_te.push_back(Fl[s]);
				}
				else if (k < Fd) {
					Fl_tr.push_back(Fl[s]);
				}
			}
		}
		for (int j = 0; j < fcase.size(); j++) {
			for (int k = 0; k < Rd; k++) {
				//デバッグの時はここを変更するべし
				//int s = j*Rd + k;
				int s = j*(fcase.size() - 2) + k;
				if ((j == i) && (k < Rd)) {
					Ref_te.push_back(Ref[s]);
				}
				else if (k < Rd) {
					Ref_tr.push_back(Ref[s]);
				}
			}
		}
		//カーネル計算
		std::vector<double> k;
		std::vector<double> Ck;
		double c;
		//kの算出
		for (int j = 0; j < n; j++) {
			double sum = 0;   //二乗和
			double ip = 0;    //内積(線形項を入れたいときはこれを使おう)
			for (int k = 0; k < Fd; k++) {
				int s = j*Fd + k;
				sum += (Fl_tr[s] - Fl_te[k])*(Fl_tr[s] - Fl_te[k]);
				ip += Fl_tr[s] * Fl_te[k];
			}
			//カーネル関数を変更するときはここを変えよう
			double ks = input_info.th3*exp(-sum / (2.0*input_info.th1*input_info.th1)) + input_info.th2*ip + input_info.th4;
			k.push_back(ks);
		}
		//C算出（β抜き）
		for (int j = 0; j < n; j++) {
			for (int m = 0; m < n; m++) {
				double sum = 0;   //二乗和
				double ip = 0;    //内積
				for (int k = 0; k < Fd; k++) {
					int s = j*Fd + k;
					int t = m*Fd + k;
					sum += (Fl_tr[s] - Fl_tr[t])*(Fl_tr[s] - Fl_tr[t]);
					ip += Fl_tr[s] * Fl_tr[t];
				}
				//カーネル関数を変更するときはここを変えよう
				double ks = input_info.th3*exp(-sum / (2.0*input_info.th1*input_info.th1)) + input_info.th2*ip + input_info.th4;
				Ck.push_back(ks);
			}
		}
		//c算出（β抜き）
		double c_ip = 0;    //内積(必要なら使う)
		for (int k = 0; k < Fd; k++) {
			c_ip += Fl_te[k] * Fl_te[k];
		}
		//カーネル関数を変更するときはここを変えよう
		//ガウシアンカーネルだとここはすべて0になる
		c = 1 / b + input_info.th2*c_ip + input_info.th4;


		//データ行列をつくる
		Eigen::MatrixXd C_k = Eigen::Map<Eigen::MatrixXd>(&Ck[0], n, n);
		Eigen::MatrixXd K = Eigen::Map<Eigen::MatrixXd>(&k[0], 1, n);      //k^T
		Eigen::MatrixXd K_ = Eigen::Map<Eigen::MatrixXd>(&k[0], n, 1);; //k
		Eigen::MatrixXd R_train = Eigen::Map<Eigen::MatrixXd>(&Ref_tr[0], Fd, n);
		Eigen::MatrixXd Y__ = Eigen::Map<Eigen::MatrixXd>(&Ref_tr[0], Rd, n);
		Eigen::MatrixXd Y = Y__.transpose();
		Eigen::MatrixXd R_train_t = R_train.transpose();       //t           
		Eigen::MatrixXd E = Eigen::MatrixXd::Identity(n, n);     //単位行列
		Eigen::MatrixXd BE = E.array() / b;
		Eigen::MatrixXd C = C_k + BE;
		Eigen::MatrixXd C_n = C.inverse();       //Cn^(-1)
		//データ行列（線形回帰用）
		Eigen::MatrixXd X_0 = Eigen::Map<Eigen::MatrixXd>(&Fl_tr[0], Fd, n);
		Eigen::MatrixXd X = X.Ones(n, Fd + 1);    //X
		X.block(0, 1, n, Fd) = X_0.transpose();
		Eigen::MatrixXd Xt_0 = Eigen::Map<Eigen::MatrixXd>(&Fl_te[0], 1, Fd);
		Eigen::MatrixXd Xt = Xt.Ones(1, Fd + 1);
		Xt.block(0, 1, 1, Fd) = Xt_0;   //Xtest
		
		//線形回帰分析
		Eigen::MatrixXd linear_0 = X.transpose()*X;
		Eigen::MatrixXd linear = linear_0.inverse()*X.transpose()*Y; //係数算出
		Eigen::MatrixXd linear_result = Xt*linear;
		//カーネル行列計算
		//平均（k^T*Cn^(-1)*t）
		Eigen::MatrixXd mean = K*C_n*R_train_t;
		//分散（c-k^T*Cn^(-1)*k）
		Eigen::MatrixXd var_ = K*C_n*K_;
		std::cout << var_ << std::endl;
		double v = var_(0, 0);
		std::cout << v << std::endl;
		double var = c - v;
		std::stringstream dirOUT;
		std::stringstream dirOUT2;
		std::stringstream dirOUT3;
		dirOUT << input_info.dir_out << fcase[i] << "/mean";
		dirOUT2 << input_info.dir_out << fcase[i] << "/var";
		dirOUT3 << input_info.dir_out << fcase[i] << "/linear";
		nari::system::make_directry(dirOUT.str());
		nari::system::make_directry(dirOUT2.str());
		write_matrix_raw_and_txt(mean, dirOUT.str());
		std::ofstream mat_result(dirOUT.str() + ".txt");
		std::ofstream mat_result2(dirOUT2.str() + ".txt");
		std::ofstream mat_result3(dirOUT3.str() + ".txt");
		for (int j = 0; j < Rd; j++) {
			mat_result << mean(0, j) << std::endl;
		}
		for (int j = 0; j < Rd; j++) {
			mat_result3 << linear_result(0, j) << std::endl;
		}
		mat_result2 << var << std::endl;
	}
}
