/*
	Visualization demo
*/

#include "generator.h"
#include "python_interop.h"

#include <sstream>
#include <chrono>
#include <random>
#include <iostream>

#include <QApplication>
#include <QtWidgets>

/*************************************************************************************************************************************/

class Demo : public QMainWindow
{
	Generator    _gan;
	Python       _py;
	LatentVector _z;

	QLabel*      _imageArea;
	QLabel*		 _imageAreaSmall;
	QLabel*		 _nearestNeighbourArea;
	QLabel*		 _nearestNeighbourAreaSmall;
	QLabel*		 _infoLabel;

	std::vector<QSlider*> _coordinates;

public:

	Demo()
	{
		// zero latent vector
		_z.fill(0);

		// left panel
		auto dock = new QDockWidget(this);
		auto container = new QWidget(dock);
		container->setMinimumWidth(200);
		auto panelLayout = new QVBoxLayout(container);

		const QString coordFormat = QStringLiteral("z[%1] = %2");

		for (size_t i = 0; i < 10; i++)
		{
			auto label = new QLabel(coordFormat.arg(QString::number(i), QString::number(0.0f)), container);
			auto coord = new QSlider(Qt::Horizontal, container);
			coord->setValue(0);
			coord->setMinimum(0);
			coord->setMaximum(1000);
			panelLayout->addWidget(label);
			panelLayout->addWidget(coord);
			_coordinates.push_back(coord);
			// slider changed by user
			QObject::connect(coord, &QSlider::sliderMoved, [i, this](int value) {
				float z = (float)value / 1000;
				std::cout << z << std::endl;
				this->_z[i] = z;
				this->updateImage();
			});
			// slider changed by anything
			QObject::connect(coord, &QSlider::valueChanged, [i, label, coordFormat](int value) {
				float z = (float)value / 1000;
				label->setText(coordFormat.arg(QString::number(i), QString::number(z)));
			});
		}

		auto randomize = new QPushButton(QStringLiteral("randomize"), container);
		QObject::connect(randomize, &QPushButton::clicked, this, &Demo::randomizeZ);
		panelLayout->addWidget(randomize);

		container->setLayout(panelLayout);
		dock->setWidget(container);
		addDockWidget(Qt::LeftDockWidgetArea, dock);

		// center
		auto center = new QWidget(this);
		auto centerLayout = new QHBoxLayout(center);
		auto fakeLayout = new QVBoxLayout(center);
		center->setLayout(centerLayout);

		_imageArea = new QLabel(QStringLiteral("none"), center);
		_infoLabel = new QLabel("abc", center);
		_imageAreaSmall = new QLabel(center);
		fakeLayout->addWidget(_imageArea);
		fakeLayout->addWidget(_infoLabel, Qt::AlignTop);
		fakeLayout->addWidget(_imageAreaSmall);
		centerLayout->addLayout(fakeLayout);

		centerLayout->addSpacing(20);

		auto realLayout = new QVBoxLayout(center);
		_nearestNeighbourArea = new QLabel(center);
		_nearestNeighbourAreaSmall = new QLabel(center);
		realLayout->addWidget(_nearestNeighbourArea);
		realLayout->addWidget(new QLabel(QStringLiteral("real sample"), center), Qt::AlignHCenter);
		realLayout->addWidget(_nearestNeighbourAreaSmall);
		centerLayout->addLayout(realLayout);

		setCentralWidget(center);
		updateImage();
	}

	void randomizeZ()
	{
		std::random_device dev;
		std::default_random_engine rng(dev());
		std::uniform_real_distribution<float> d(0, 1);
		std::generate(_z.begin(), _z.end(), [&](){ return d(rng); });
		for (size_t i = 0; i < 10; i++)
		{
			_coordinates[i]->setValue(_z[i] * 1000);
		}
		updateImage();
	}

	void setZ(const LatentVector& z)
	{
		_z = z;
		updateImage();
	}

	void updateImage()
	{
		using Time = std::chrono::high_resolution_clock;
		using std::chrono::duration_cast;
		using std::chrono::microseconds;

		QSize size(28, 28);

		ImageOutput img;
		ImageOutput real;

		auto t = Time::now();

		// generate image
		_gan.generate(_z, img);
		float model_elapsed = (float)duration_cast<microseconds>(Time::now() - t).count() / (1000); //in milliseconds

		_py.nearest_neighbour(img, real);
		
		float elapsed = (float)duration_cast<microseconds>(Time::now() - t).count() / (1000); //in milliseconds

		// process image ([0..1] -> [0..255])
		std::array<uchar, 28 * 28> img_data;
		std::transform(img.begin(), img.end(), img_data.begin(), [](float i) { return (uchar)(i * 255); });

		// update pixmap
		QImage qimg(img_data.data(), size.width(), size.height(), QImage::Format_Grayscale8);
		qimg = qimg.scaled(size * 12, Qt::KeepAspectRatio);
		_imageArea->setPixmap(QPixmap::fromImage(qimg));
		// update label
		_infoLabel->setText(QString("resolution (28x28)\nexecution time: %1ms\nupdate time: %2ms").arg(QString::number(model_elapsed), QString::number(elapsed)));
		_infoLabel->setWordWrap(true);

		std::transform(real.begin(), real.end(), img_data.begin(), [](float i) { return (uchar)(i * 255); });
		QImage qimg2(img_data.data(), size.width(), size.height(), QImage::Format_Grayscale8);
		qimg2 = qimg2.scaled(size * 12, Qt::KeepAspectRatio);
		_nearestNeighbourArea->setPixmap(QPixmap::fromImage(qimg2));
	}
};

/*************************************************************************************************************************************/

int main(int argc, char** argv)
{
	QApplication app(argc, argv);
	QFont font = app.font();
	font.setPointSize(10);
	app.setFont(font);

	app.setStyle(QStyleFactory::create("Fusion"));
	QPalette darkPalette;
	QColor darkColor = QColor(45, 45, 45);
	QColor disabledColor = QColor(127, 127, 127);
	darkPalette.setColor(QPalette::Window, darkColor);
	darkPalette.setColor(QPalette::WindowText, Qt::white);
	darkPalette.setColor(QPalette::Base, QColor(18, 18, 18));
	darkPalette.setColor(QPalette::AlternateBase, darkColor);
	darkPalette.setColor(QPalette::ToolTipBase, Qt::white);
	darkPalette.setColor(QPalette::ToolTipText, Qt::white);
	darkPalette.setColor(QPalette::Text, Qt::white);
	darkPalette.setColor(QPalette::Disabled, QPalette::Text, disabledColor);
	darkPalette.setColor(QPalette::Button, darkColor);
	darkPalette.setColor(QPalette::ButtonText, Qt::white);
	darkPalette.setColor(QPalette::Disabled, QPalette::ButtonText, disabledColor);
	darkPalette.setColor(QPalette::BrightText, Qt::red);
	darkPalette.setColor(QPalette::Link, QColor(42, 130, 218));

	darkPalette.setColor(QPalette::Highlight, QColor(42, 130, 218));
	darkPalette.setColor(QPalette::HighlightedText, Qt::black);
	darkPalette.setColor(QPalette::Disabled, QPalette::HighlightedText, disabledColor);

	app.setPalette(darkPalette);

	Demo win;
	win.show();

	return app.exec();
}

/*************************************************************************************************************************************/
