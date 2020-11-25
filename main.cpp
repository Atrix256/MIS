#include <random>
#include <vector>

#define DETERMINISTIC() true
#define DO_BLUE_NOISE() false  // blue noise is very slow to generate, and doesn't have good convergence speed

#define DO_PDF_TEST() true

static const size_t c_numSamples = 5000;
static const size_t c_numTests = 10000;

static const double c_pi = 3.14159265359;
static const double c_goldeRatioConjugate = 0.61803398875;
static const double c_sqrt2 = sqrt(2.0);
static const double c_minError = 0.00001; // to avoid errors when showing on a log plot

std::mt19937 GetRNG(int seed)
{
#if DETERMINISTIC()
    static int internalSeed = 0;
    internalSeed++;
    std::mt19937 mt(internalSeed * 1000 + seed);
#else
    std::random_device rd("/dev/random");
    std::mt19937 mt(rd());
#endif
    return mt;
}

struct Result
{
    double estimate;
    std::vector<double> estimates;

    double estimateAvg;
    std::vector<double> estimatesAvg;

    double estimateSqAvg;
    std::vector<double> estimatesSqAvg;
};

inline double max(double a, double b)
{
    return a >= b ? a : b;
}

inline double min(double a, double b)
{
    return a <= b ? a : b;
}

inline double TorroidalDistance(double a, double b)
{
    double dist = abs(a - b);
    if (dist > 0.5)
        dist = 1.0 - dist;
    return dist;
}

void MakeBlueNoise(std::vector<double>& samples, size_t sampleCount, int seed)
{
#if DO_BLUE_NOISE()
    std::mt19937 rng = GetRNG(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    samples.resize(sampleCount);
    for (size_t sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex)
    {
        double bestCandidate = 0.0f;
        double bestCandidateScore = 0.0f;

        for (size_t candidateIndex = 0; candidateIndex <= sampleIndex; ++candidateIndex)
        {
            double candidate = dist(rng);
            double minDist = FLT_MAX;
            for (size_t distIndex = 0; distIndex < sampleIndex; ++distIndex)
                minDist = min(minDist, TorroidalDistance(candidate, samples[distIndex]));

            if (minDist > bestCandidateScore)
            {
                bestCandidate = candidate;
                bestCandidateScore = minDist;
            }
        }

        samples[sampleIndex] = bestCandidate;
    }
#else
    std::mt19937 rng = GetRNG(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    samples.resize(sampleCount);
    for (double& d : samples)
        d = dist(rng);
#endif
}

template <typename T>
T Clamp(T value, T min, T max)
{
    if (value <= min)
        return min;
    if (value >= max)
        return max;
    return value;
}

double Lerp(double A, double B, double t)
{
    return A * (1.0 - t) + B * t;
}

void AddSampleToRunningAverage(double &average, double newValue, size_t sampleCount)
{
    // Incremental averaging: lerp from old value to new value by 1/(sampleCount+1)
    // https://blog.demofox.org/2016/08/23/incremental-averaging/
    double t = 1.0 / double(sampleCount + 1);
    average = Lerp(average, newValue, t);
}

template <typename TF>
void MonteCarlo(const TF& F, Result& result, int seed)
{
    std::mt19937 rng = GetRNG(seed);
    std::uniform_real_distribution<double> dist(0.0, c_pi);

    result.estimates.resize(c_numSamples);

    result.estimate = 0.0f;
    for (size_t i = 0; i < c_numSamples; ++i)
    {
        double x = dist(rng);
        double y = F(x);
        AddSampleToRunningAverage(result.estimate, y, i);
        result.estimates[i] = result.estimate * c_pi;
    }

    result.estimate *= c_pi;
}

template <typename TF>
void MonteCarloLDS(const TF& F, Result& result, int seed)
{
    std::mt19937 rng = GetRNG(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0f);

    result.estimates.resize(c_numSamples);
    double lds = dist(rng);
    result.estimate = 0.0f;
    for (size_t i = 0; i < c_numSamples; ++i)
    {
        double x = lds * c_pi;
        double y = F(x);
        AddSampleToRunningAverage(result.estimate, y, i);
        result.estimates[i] = result.estimate * c_pi;
        lds = fmod(lds + c_goldeRatioConjugate, 1.0f);
    }

    result.estimate *= c_pi;
}

template <typename TF>
void MonteCarloBlue(const TF& F, Result& result, int seed)
{
    std::vector<double> blueNoise;
    MakeBlueNoise(blueNoise, c_numSamples, seed);

    result.estimates.resize(c_numSamples);
    result.estimate = 0.0f;
    for (size_t i = 0; i < c_numSamples; ++i)
    {
        double x = blueNoise[i] * c_pi;
        double y = F(x);
        AddSampleToRunningAverage(result.estimate, y, i);
        result.estimates[i] = result.estimate * c_pi;
    }

    result.estimate *= c_pi;
}

template <typename TF, typename TPDF, typename TINVERSECDF>
void ImportanceSampledMonteCarlo(const TF& F, const TPDF& PDF, const TINVERSECDF& InverseCDF, Result& result, int seed)
{
    std::mt19937 rng = GetRNG(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    result.estimates.resize(c_numSamples);

    result.estimate = 0.0f;
    for (size_t i = 0; i < c_numSamples; ++i)
    {
        double x = InverseCDF(dist(rng));
        double y = F(x);
        double pdf = PDF(x);
        double value = y / pdf;
        AddSampleToRunningAverage(result.estimate, value, i);
        result.estimates[i] = result.estimate;
    }
}

template <typename TF, typename TPDF, typename TINVERSECDF>
void ImportanceSampledMonteCarloLDS(const TF& F, const TPDF& PDF, const TINVERSECDF& InverseCDF, Result& result, int seed)
{
    std::mt19937 rng = GetRNG(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0f);

    result.estimates.resize(c_numSamples);
    double lds = dist(rng);
    result.estimate = 0.0f;
    for (size_t i = 0; i < c_numSamples; ++i)
    {
        double x = InverseCDF(lds);
        double y = F(x);
        double pdf = PDF(x);
        double value = y / pdf;
        AddSampleToRunningAverage(result.estimate, value, i);
        result.estimates[i] = result.estimate;
        lds = fmod(lds + c_goldeRatioConjugate, 1.0f);
    }
}

template <typename TF, typename TPDF, typename TINVERSECDF>
void ImportanceSampledMonteCarloBlue(const TF& F, const TPDF& PDF, const TINVERSECDF& InverseCDF, Result& result, int seed)
{
    std::vector<double> blueNoise;
    MakeBlueNoise(blueNoise, c_numSamples, seed);

    result.estimates.resize(c_numSamples);
    result.estimate = 0.0f;
    for (size_t i = 0; i < c_numSamples; ++i)
    {
        double x = InverseCDF(blueNoise[i]);
        double y = F(x);
        double pdf = PDF(x);
        double value = y / pdf;
        AddSampleToRunningAverage(result.estimate, value, i);
        result.estimates[i] = result.estimate;
    }
}

template <typename TF, typename TPDF1, typename TINVERSECDF1, typename TPDF2, typename TINVERSECDF2>
void MultipleImportanceSampledMonteCarlo(const TF& F, const TPDF1& PDF1, const TINVERSECDF1& InverseCDF1, const TPDF2& PDF2, const TINVERSECDF2& InverseCDF2, Result& result, int seed)
{
    std::mt19937 rng = GetRNG(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    result.estimates.resize(c_numSamples);

    result.estimate = 0.0f;
    for (size_t i = 0; i < c_numSamples; ++i)
    {
        double x1 = InverseCDF1(dist(rng));
        double y1 = F(x1);
        double pdf11 = PDF1(x1);
        double pdf12 = PDF2(x1);

        double x2 = InverseCDF2(dist(rng));
        double y2 = F(x2);
        double pdf21 = PDF1(x2);
        double pdf22 = PDF2(x2);

        double value =
            y1 / (pdf11 + pdf12) +
            y2 / (pdf21 + pdf22)
        ;

        AddSampleToRunningAverage(result.estimate, value, i);
        result.estimates[i] = result.estimate;
    }
}

template <typename TF, typename TPDF1, typename TINVERSECDF1, typename TPDF2, typename TINVERSECDF2, typename TPDF3, typename TINVERSECDF3>
void MultipleImportanceSampledMonteCarlo(const TF& F, const TPDF1& PDF1, const TINVERSECDF1& InverseCDF1, const TPDF2& PDF2, const TINVERSECDF2& InverseCDF2, const TPDF3& PDF3, const TINVERSECDF3& InverseCDF3, Result& result, int seed)
{
    std::mt19937 rng = GetRNG(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    result.estimates.resize(c_numSamples);

    result.estimate = 0.0f;
    for (size_t i = 0; i < c_numSamples; ++i)
    {
        double x1 = InverseCDF1(dist(rng));
        double y1 = F(x1);
        double pdf11 = PDF1(x1);
        double pdf12 = PDF2(x1);
        double pdf13 = PDF3(x1);

        double x2 = InverseCDF2(dist(rng));
        double y2 = F(x2);
        double pdf21 = PDF1(x2);
        double pdf22 = PDF2(x2);
        double pdf23 = PDF3(x2);

        double x3 = InverseCDF3(dist(rng));
        double y3 = F(x3);
        double pdf31 = PDF1(x3);
        double pdf32 = PDF2(x3);
        double pdf33 = PDF3(x3);

        double value =
            y1 / (pdf11 + pdf12 + pdf13) +
            y2 / (pdf21 + pdf22 + pdf23) + 
            y3 / (pdf31 + pdf32 + pdf33)
            ;

        AddSampleToRunningAverage(result.estimate, value, i);
        result.estimates[i] = result.estimate;
    }
}

template <typename TF, typename TPDF1, typename TINVERSECDF1, typename TPDF2, typename TINVERSECDF2>
void MultipleImportanceSampledMonteCarloStochastic(const TF& F, const TPDF1& PDF1, const TINVERSECDF1& InverseCDF1, const TPDF2& PDF2, const TINVERSECDF2& InverseCDF2, Result& result, int seed)
{
    std::mt19937 rng = GetRNG(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    result.estimates.resize(c_numSamples);

    result.estimate = 0.0f;
    for (size_t i = 0; i < c_numSamples; ++i)
    {
        double x1 = InverseCDF1(dist(rng));
        double pdf11 = PDF1(x1);
        double pdf12 = PDF2(x1);
        double weight1 = pdf11 / (pdf11 + pdf12);

        double x2 = InverseCDF2(dist(rng));
        double pdf21 = PDF1(x2);
        double pdf22 = PDF2(x2);
        double weight2 = pdf22 / (pdf21 + pdf22);

        double totalWeight = weight1 + weight2;

        double weight1chance = weight1 / totalWeight;

        double estimate = dist(rng) < weight1chance
            ? (F(x1) / pdf11) * totalWeight  // (F(x1) / pdf11) * (weight1 / weight1chance)
            : (F(x2) / pdf22) * totalWeight; // (F(x2) / pdf22) * (weight2 / (1.0 - weight1chance));

        AddSampleToRunningAverage(result.estimate, estimate, i);
        result.estimates[i] = result.estimate;
    }
}

template <typename TF, typename TPDF1, typename TINVERSECDF1, typename TPDF2, typename TINVERSECDF2>
void MultipleImportanceSampledMonteCarloLDS(const TF& F, const TPDF1& PDF1, const TINVERSECDF1& InverseCDF1, const TPDF2& PDF2, const TINVERSECDF2& InverseCDF2, Result& result, int seed)
{
    std::mt19937 rng = GetRNG(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    result.estimates.resize(c_numSamples);

    double lds1 = dist(rng);
    double lds2 = dist(rng);
    result.estimate = 0.0f;
    for (size_t i = 0; i < c_numSamples; ++i)
    {
        double x1 = InverseCDF1(lds1);
        double y1 = F(x1);
        double pdf11 = PDF1(x1);
        double pdf12 = PDF2(x1);

        double x2 = InverseCDF2(lds2);
        double y2 = F(x2);
        double pdf21 = PDF1(x2);
        double pdf22 = PDF2(x2);

        double value =
            y1 / (pdf11 + pdf12) +
            y2 / (pdf21 + pdf22)
            ;

        AddSampleToRunningAverage(result.estimate, value, i);
        result.estimates[i] = result.estimate;

        lds1 = fmod(lds1 + c_goldeRatioConjugate, 1.0f);
        lds2 = fmod(lds2 + c_sqrt2, 1.0f);
    }
}

void IntegrateResult(Result& sample, int sampleCount)
{
    AddSampleToRunningAverage(sample.estimateAvg, sample.estimate, sampleCount);
    AddSampleToRunningAverage(sample.estimateSqAvg, sample.estimate * sample.estimate, sampleCount);

    sample.estimatesAvg.resize(sample.estimates.size());
    sample.estimatesSqAvg.resize(sample.estimates.size());

    for (size_t index = 0; index < sample.estimates.size(); ++index)
    {
        AddSampleToRunningAverage(sample.estimatesAvg[index], sample.estimates[index], sampleCount);
        AddSampleToRunningAverage(sample.estimatesSqAvg[index], sample.estimates[index] * sample.estimates[index], sampleCount);
    }
}

void PDFTest1()
{
    std::mt19937 rng = GetRNG(0);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    double sumAverage = 0.0;

    // y=x normalized to y=x * 2
    auto PDF1 = [](double x) -> double
    {
        return x * 2.0;
    };

    // y=1
    auto PDF2 = [](double x) -> double
    {
        return 1.0;
    };

    // y=sin(x*pi) normalized to y=sin(x*pi)*pi/2
    auto PDF3 = [](double x) -> double
    {
        return sin(x * c_pi) * c_pi / 2.0;
    };

    for (size_t index = 0; index < 100000; ++index)
    {
        double x = dist(rng);
        double sum = PDF1(x) + PDF2(x) + PDF3(x);
        AddSampleToRunningAverage(sumAverage, sum, index);
    }

    printf("PDFTest1: %f (%f)\n", sumAverage, sumAverage - 3.0 / 1.0);
}

void PDFTest2()
{
    std::mt19937 rng = GetRNG(0);
    std::uniform_real_distribution<double> dist(0.0, 5.0);

    double sumAverage = 0.0;
    //double OOSumAverage = 0.0;

    // y=x normalized to y=x*2/25
    auto PDF1 = [](double x) -> double
    {
        return x * 2.0/25.0;
    };

    // y=1 normalized to y=1/5
    auto PDF2 = [](double x) -> double
    {
        return 1.0 / 5.0;
    };

    // y=sin(x*pi/5) normalized to y=sin(x*pi/5) * pi/10
    auto PDF3 = [](double x) -> double
    {
        return sin(x * c_pi / 5.0) * c_pi / 10.0;
    };

    for (size_t index = 0; index < 100000; ++index)
    {
        double x = dist(rng);
        double sum = PDF1(x) + PDF2(x) + PDF3(x);
        AddSampleToRunningAverage(sumAverage, sum, index);
    }

    printf("PDFTest2: %f (%f)\n", sumAverage, sumAverage - 3.0 / 5.0);
}

void PDFTest3()
{
    std::mt19937 rng = GetRNG(0);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // The function we are integrating
    auto F = [](double x) -> double
    {
        return sin(x) * 2.0 * x;
    };

    // the PDF and inverse CDF of distributions we are using for integration
    auto PDF1 = [](double x) -> double
    {
        // normalizing y=sin(x) from 0 to pi to integrate to 1 from 0 to pi
        return sin(x) / 2.0;
    };

    auto InverseCDF1 = [](double x) -> double
    {
        // turning the PDF into a CDF, flipping x and y, and solving for y again
        return 2.0 * asin(sqrt(x));
    };

    auto PDF2 = [](double x) -> double
    {
        // normalizing y=2x from 0 to pi to integrate to 1 from 0 to pi
        return x * 2.0 / (c_pi * c_pi);
    };

    auto InverseCDF2 = [](double x) -> double
    {
        // turning the PDF into a CDF, flipping x and y, and solving for y again
        return c_pi * sqrt(x);
    };

    double c_actual = 2.0 * c_pi;

    double integrated1 = 0.0;
    double integrated2 = 0.0;
    double integrated = 0.0;

    for (size_t index = 0; index < 100000; ++index)
    {
        double rng1 = dist(rng);
        double x1 = InverseCDF1(rng1);
        double y1 = F(x1) / (PDF1(x1) + PDF2(x1));

        double rng2 = dist(rng);
        double x2 = InverseCDF2(dist(rng));
        double y2 = F(x2) / (PDF1(x2) + PDF2(x2));

        AddSampleToRunningAverage(integrated1, y1, index);
        AddSampleToRunningAverage(integrated2, y2, index);
        AddSampleToRunningAverage(integrated, y1 + y2, index);
    }

    printf("PDFTest3   A: %f (%f)\n", integrated1, integrated1 - c_actual);
    printf("PDFTest3   B: %f (%f)\n", integrated2, integrated2 - c_actual);
    printf("PDFTest3 A+B: %f (%f)\n", integrated, integrated - c_actual);
}

void PrintfResult(const char* label, const Result& result, double actual)
{
    double variance = abs(result.estimateSqAvg - (result.estimateAvg * result.estimateAvg));
    printf("  %s = %f | abse %f | var %f\n", label, result.estimateAvg, abs(result.estimateAvg - actual), variance);
}

template <typename TPDF>
void TestPDF(const TPDF& PDF, double min, double max, const char* fileName)
{
    static const size_t c_buckets = 100;
    FILE* file = nullptr;
    fopen_s(&file, fileName, "w+b");
    fprintf(file, "\"x\",\"PDF(x)\"\n");
    for (size_t index = 0; index < c_buckets; ++index)
    {
        double percent = (double(index) + 0.5) / double(c_buckets);
        double x = Lerp(min, max, percent);
        double pdf = PDF(x);
        fprintf(file, "\"%f\",\"%f\"\n", x, pdf);
    }
    fclose(file);
}

template <typename TICDF>
void TestICDF(const TICDF& ICDF, double min, double max, const char* fileName)
{
    static const size_t c_buckets = 100;
    static const size_t c_values = 10000;

    std::vector<size_t> histogram(c_buckets, 0);

    for (size_t index = 0; index < c_values; ++index)
    {
        double percent = (double(index) + 0.5) / double(c_values);
        double x = ICDF(percent);

        double xpercent = (x - min) / (max - min);
        size_t xbucket = Clamp<size_t>(size_t(xpercent * double(c_buckets)), 0, c_buckets - 1);
        histogram[xbucket]++;
    }

    FILE* file = nullptr;
    fopen_s(&file, fileName, "w+b");
    fprintf(file, "\"x\",\"count\"\n");
    for (size_t index = 0; index < c_buckets; ++index)
    {
        double percent = (double(index) + 0.5) / double(c_values);
        fprintf(file, "\"%f\",\"%zu\"\n", percent, histogram[index]);
    }

    fclose(file);
}

int main(int argc, char** argv)
{
#if DO_PDF_TEST() // experiments to help demonstrate how and why MIS works
    printf("PDF tests to show aspects of MIS\n");
    PDFTest1(); // showing how the expected value of the sum of 3 PDF's is 3 (instead of 1, it's from 0 to 1)
    PDFTest2(); // showing how the expected value of the sum of 3 PDS's are 3x too large. (0.6 instead of 0.2, it's from 0 to 5)
    PDFTest3(); // showing how this MIS formulaion with 2 PDFs makes two values which are each 50% sized, so adding them together makes the correct value.
    printf("\n\n\n");
#endif

    printf("%zu tests, with %zu samples each.\n\n", c_numTests, c_numSamples);

    // y = sin(x)*sin(x) from 0 to pi
    {
        Result mc;
        Result mcblue;
        Result mclds;
        Result ismc;
        Result ismcblue;
        Result ismclds;

        // The function we are integrating
        auto F = [](double x) -> double
        {
            return sin(x) * sin(x);
        };

        // the PDF and inverse CDF of a distribution we are using for integration
        auto PDF = [](double x) -> double
        {
            // normalizing y=sin(x) from 0 to pi to integrate to 1 from 0 to pi
            return sin(x) / 2.0;
        };

        auto InverseCDF = [](double x) -> double
        {
            // turning the PDF into a CDF, flipping x and y, and solving for y again
            return 2.0 * asin(sqrt(x));
        };

        double c_actual = c_pi / 2.0;

        // numerical integration
        for (int testIndex = 0; testIndex < c_numTests; ++testIndex)
        {
            MonteCarlo(F, mc, testIndex);
            MonteCarloBlue(F, mcblue, testIndex);
            MonteCarloLDS(F, mclds, testIndex);
            ImportanceSampledMonteCarlo(F, PDF, InverseCDF, ismc, testIndex);
            ImportanceSampledMonteCarloBlue(F, PDF, InverseCDF, ismcblue, testIndex);
            ImportanceSampledMonteCarloLDS(F, PDF, InverseCDF, ismclds, testIndex);

            IntegrateResult(mc, testIndex);
            IntegrateResult(mcblue, testIndex);
            IntegrateResult(mclds, testIndex);
            IntegrateResult(ismc, testIndex);
            IntegrateResult(ismcblue, testIndex);
            IntegrateResult(ismclds, testIndex);
        }

        // report results
        {
            // summary to screen
            printf("y = sin(x)*sin(x) from 0 to pi\n");
            PrintfResult("mc       ", mc, c_actual);
            PrintfResult("mcblue   ", mcblue, c_actual);
            PrintfResult("mclds    ", mclds, c_actual);
            PrintfResult("ismc     ", ismc, c_actual);
            PrintfResult("ismcblue ", ismcblue, c_actual);
            PrintfResult("ismclds  ", ismclds, c_actual);
            printf("\n");

            // details to csv
            FILE* file = nullptr;
            fopen_s(&file, "out1.csv", "wb");
            fprintf(file, "\"index\",\"mc\",\"mcblue\",\"mclds\",\"ismc\",\"ismcblue\",\"ismclds\"\n");
            for (size_t i = 0; i < c_numSamples; ++i)
            {
                fprintf(file, "\"%zu\",", i);
                fprintf(file, "\"%f\",", max(abs(mc.estimatesAvg[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(mcblue.estimatesAvg[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(mclds.estimatesAvg[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(ismc.estimatesAvg[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(ismcblue.estimatesAvg[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(ismclds.estimatesAvg[i] - c_actual), c_minError));
                fprintf(file, "\n");
            }
            fclose(file);
        }
    }

    // y=sin(x)*2x from 0 to pi
    {
        Result mc;
        Result mcblue;
        Result mclds;
        Result ismc1;
        Result ismcblue1;
        Result ismclds1;
        Result ismc2;
        Result ismcblue2;
        Result ismclds2;
        Result mismc;
        Result mismclds;
        Result mismcstoc;

        // The function we are integrating
        auto F = [](double x) -> double
        {
            return sin(x) * 2.0 * x;
        };

        // the PDF and inverse CDF of distributions we are using for integration
        auto PDF1 = [](double x) -> double
        {
            // normalizing y=sin(x) from 0 to pi to integrate to 1 from 0 to pi
            return sin(x) / 2.0;
        };

        auto InverseCDF1 = [](double x) -> double
        {
            // turning the PDF into a CDF, flipping x and y, and solving for y again
            return 2.0 * asin(sqrt(x));
        };

        auto PDF2 = [](double x) -> double
        {
            // normalizing y=2x from 0 to pi to integrate to 1 from 0 to pi
            return x * 2.0 / (c_pi * c_pi);
        };

        auto InverseCDF2 = [](double x) -> double
        {
            // turning the PDF into a CDF, flipping x and y, and solving for y again
            return c_pi * sqrt(x);
        };

        double c_actual = 2.0 * c_pi;

        // numerical integration
        for (int testIndex = 0; testIndex < c_numTests; ++testIndex)
        {
            MonteCarlo(F, mc, testIndex);
            MonteCarloBlue(F, mcblue, testIndex);
            MonteCarloLDS(F, mclds, testIndex);
            ImportanceSampledMonteCarlo(F, PDF1, InverseCDF1, ismc1, testIndex);
            ImportanceSampledMonteCarloBlue(F, PDF1, InverseCDF1, ismcblue1, testIndex);
            ImportanceSampledMonteCarloLDS(F, PDF1, InverseCDF1, ismclds1, testIndex);
            ImportanceSampledMonteCarlo(F, PDF2, InverseCDF2, ismc2, testIndex);
            ImportanceSampledMonteCarloBlue(F, PDF2, InverseCDF2, ismcblue2, testIndex);
            ImportanceSampledMonteCarloLDS(F, PDF2, InverseCDF2, ismclds2, testIndex);
            MultipleImportanceSampledMonteCarlo(F, PDF1, InverseCDF1, PDF2, InverseCDF2, mismc, testIndex);
            MultipleImportanceSampledMonteCarloLDS(F, PDF1, InverseCDF1, PDF2, InverseCDF2, mismclds, testIndex);
            MultipleImportanceSampledMonteCarloStochastic(F, PDF1, InverseCDF1, PDF2, InverseCDF2, mismcstoc, testIndex);

            IntegrateResult(mc, testIndex);
            IntegrateResult(mcblue, testIndex);
            IntegrateResult(mclds, testIndex);
            IntegrateResult(ismc1, testIndex);
            IntegrateResult(ismcblue1, testIndex);
            IntegrateResult(ismclds1, testIndex);
            IntegrateResult(ismc2, testIndex);
            IntegrateResult(ismcblue2, testIndex);
            IntegrateResult(ismclds2, testIndex);
            IntegrateResult(mismc, testIndex);
            IntegrateResult(mismclds, testIndex);
            IntegrateResult(mismcstoc, testIndex);
        }

        // report results
        {
            // summary to screen
            printf("y = sin(x)*2x from 0 to pi\n");
            PrintfResult("mc       ", mc, c_actual);
            PrintfResult("mcblue   ", mcblue, c_actual);
            PrintfResult("mclds    ", mclds, c_actual);
            PrintfResult("ismc1    ", ismc1, c_actual);
            PrintfResult("ismcblue1", ismcblue1, c_actual);
            PrintfResult("ismclds1 ", ismclds1, c_actual);
            PrintfResult("ismc2    ", ismc2, c_actual);
            PrintfResult("ismcblue2", ismcblue2, c_actual);
            PrintfResult("ismclds2 ", ismclds2, c_actual);
            PrintfResult("mismc    ", mismc, c_actual);
            PrintfResult("mismclds ", mismclds, c_actual);
            PrintfResult("mismcstoc", mismcstoc, c_actual);
            printf("\n");

            // details to csv
            FILE* file = nullptr;
            fopen_s(&file, "out2.csv", "wb");
            fprintf(file, "\"index\",\"mc\",\"mcblue\",\"mclds\",\"ismc1\",\"ismcblue1\",\"ismclds1\",\"ismc2\",\"ismcblue2\",\"ismclds2\",\"mismc\",\"mismcstoc\",\"mismclds\"\n");
            for (size_t i = 0; i < c_numSamples; ++i)
            {
                fprintf(file, "\"%zu\",", i);
                fprintf(file, "\"%f\",", max(abs(mc.estimatesAvg[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(mcblue.estimatesAvg[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(mclds.estimatesAvg[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(ismc1.estimatesAvg[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(ismcblue1.estimatesAvg[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(ismclds1.estimatesAvg[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(ismc2.estimatesAvg[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(ismcblue2.estimatesAvg[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(ismclds2.estimatesAvg[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(mismc.estimatesAvg[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(mismclds.estimatesAvg[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(mismcstoc.estimatesAvg[i] - c_actual), c_minError));
                fprintf(file, "\n");
            }
            fclose(file);
        }
    }

    // y=sin(x*3)*sin(x*3)*sin(x)*sin(x) from 0 to pi
    {
        Result mc;
        Result mismc;

        // The function we are integrating
        auto F = [](double x) -> double
        {
            return sin(x * 3.0) * sin(x * 3.0) * sin(x) * sin(x);
        };

        // the PDF and inverse CDF of distributions we are using for integration
        auto PDF = [](double x) -> double
        {
            // normalizing y=sin(x) from 0 to pi to integrate to 1 from 0 to pi
            return sin(x) / 2.0;
        };

        auto InverseCDF = [](double x) -> double
        {
            // turning the PDF into a CDF, flipping x and y, and solving for y again
            return 2.0 * asin(sqrt(x));
        };

        // [0,1/3)
        auto PDF1 = [PDF](double x) -> double
        {
            x = x * 3.0 - 0.0 * c_pi;

            if (x < 0.0 || x >= c_pi)
                return 0.0;

            return PDF(x) * 3.0;
        };

        auto InverseCDF1 = [InverseCDF](double x) -> double
        {
            return InverseCDF(x) * 1.0 / 3.0 + c_pi * 0.0 / 3.0;
        };

        // [1/3,2/3)
        auto PDF2 = [PDF](double x) -> double
        {
            x = x * 3.0 - 1.0 * c_pi;

            if (x < 0.0 || x >= c_pi)
                return 0.0;

            return PDF(x) * 3.0;
        };

        auto InverseCDF2 = [InverseCDF](double x) -> double
        {
            return InverseCDF(x) * 1.0 / 3.0 + c_pi * 1.0 / 3.0;
        };

        // [2/3,2/3]
        auto PDF3 = [PDF](double x) -> double
        {
            x = x * 3.0 - 2.0 * c_pi;

            if (x < 0.0 || x > c_pi)
                return 0.0;

            return PDF(x) * 3.0;
        };

        auto InverseCDF3 = [InverseCDF](double x) -> double
        {
            return InverseCDF(x) * 1.0 / 3.0 + c_pi * 2.0 / 3.0;
        };

        TestPDF(PDF1, 0.0, c_pi, "out3_pdf1.csv");
        TestPDF(PDF2, 0.0, c_pi, "out3_pdf2.csv");
        TestPDF(PDF3, 0.0, c_pi, "out3_pdf3.csv");
        TestICDF(InverseCDF1, 0.0, c_pi, "out3_icdf1.csv");
        TestICDF(InverseCDF2, 0.0, c_pi, "out3_icdf2.csv");
        TestICDF(InverseCDF3, 0.0, c_pi, "out3_icdf3.csv");

        double c_actual = c_pi / 4.0;

        // numerical integration
        for (int testIndex = 0; testIndex < c_numTests; ++testIndex)
        {
            MonteCarlo(F, mc, testIndex);
            MultipleImportanceSampledMonteCarlo(F, PDF1, InverseCDF1, PDF2, InverseCDF2, PDF3, InverseCDF3, mismc, testIndex);

            IntegrateResult(mc, testIndex);
            IntegrateResult(mismc, testIndex);
        }

        // report results
        {
            // summary to screen
            printf("y=sin(x*3)*sin(x*3)*sin(x)*sin(x) from 0 to pi\n");
            PrintfResult("mc       ", mc, c_actual);
            PrintfResult("mismc    ", mismc, c_actual);
            printf("\n");

            // details to csv
            FILE* file = nullptr;
            fopen_s(&file, "out3.csv", "wb");
            fprintf(file, "\"index\",\"mc\",\"mismc\"\n");
            for (size_t i = 0; i < c_numSamples; ++i)
            {
                fprintf(file, "\"%zu\",", i);
                fprintf(file, "\"%f\",", max(abs(mc.estimatesAvg[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(mismc.estimatesAvg[i] - c_actual), c_minError));
                fprintf(file, "\n");
            }
            fclose(file);
        }
    }

    return 0;
}

/*

TODO:
* make variance graphs?

Blog:
 * show error on log/log: scatter plot in open office, then format x/y axis to log
 * if no bias, reducing variance is the same as removing error.
 ? should we show variance graphs or only error?
 * one sample mis with LDS for stochastic choice should be better than rng. find a good 3d lds and use LDS for all parts would be interesting to look at.
 ! mis decreases variance

*/