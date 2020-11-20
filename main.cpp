#include <random>
#include <vector>

#define DETERMINISTIC() true

static const size_t c_numSamples = 500; // TODO: larger sample count?

static const double c_pi = 3.14159265359;
static const double c_goldeRatioConjugate = 0.61803398875;
static const double c_minError = 0.00001; // to avoid errors when showing on a log plot

std::mt19937 GetRNG()
{
#if DETERMINISTIC()
    std::mt19937 mt;
#else
    std::random_device rd;
    std::mt19937 mt(rd());
#endif
    return mt;
}

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

void MakeBlueNoise(std::vector<double>& samples, size_t sampleCount)
{
    std::mt19937 rng = GetRNG();
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
}

void AddSampleToRunningAverage(double &average, double newValue, size_t sampleCount)
{
    // Incremental averaging: lerp from old value to new value by 1/(sampleCount+1)
    // https://blog.demofox.org/2016/08/23/incremental-averaging/
    double t = 1.0 / double(sampleCount + 1);
    average = average * (1.0 - t) + newValue * t;
}

template <typename TF>
double MonteCarlo(const TF& F, std::vector<double>& estimates)
{
    std::mt19937 rng = GetRNG();
    std::uniform_real_distribution<double> dist(0.0, c_pi);

    estimates.resize(c_numSamples);

    double estimate = 0.0f;
    for (size_t i = 0; i < c_numSamples; ++i)
    {
        double x = dist(rng);
        double y = F(x);
        AddSampleToRunningAverage(estimate, y, i);
        estimates[i] = estimate * c_pi;
    }

    return estimate * c_pi;
}

template <typename TF>
double MonteCarloLDS(const TF& F, std::vector<double>& estimates)
{
    estimates.resize(c_numSamples);
    double lds = 0.5;
    double estimate = 0.0f;
    for (size_t i = 0; i < c_numSamples; ++i)
    {
        double x = lds * c_pi;
        double y = F(x);
        AddSampleToRunningAverage(estimate, y, i);
        estimates[i] = estimate * c_pi;
        lds = fmod(lds + c_goldeRatioConjugate, 1.0f);
    }

    return estimate * c_pi;
}

template <typename TF>
double MonteCarloBlue(const TF& F, std::vector<double>& estimates)
{
    std::vector<double> blueNoise;
    MakeBlueNoise(blueNoise, c_numSamples);

    estimates.resize(c_numSamples);
    double estimate = 0.0f;
    for (size_t i = 0; i < c_numSamples; ++i)
    {
        double x = blueNoise[i] * c_pi;
        double y = F(x);
        AddSampleToRunningAverage(estimate, y, i);
        estimates[i] = estimate * c_pi;
    }

    return estimate * c_pi;
}

template <typename TF, typename TPDF, typename TINVERSECDF>
double ImportanceSampledMonteCarlo(const TF& F, const TPDF& PDF, const TINVERSECDF& InverseCDF, std::vector<double>& estimates)
{
    std::mt19937 rng = GetRNG();
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    estimates.resize(c_numSamples);

    double estimate = 0.0f;
    for (size_t i = 0; i < c_numSamples; ++i)
    {
        double x = InverseCDF(dist(rng));
        double y = F(x);
        double pdf = PDF(x);
        double value = y / pdf;
        AddSampleToRunningAverage(estimate, value, i);
        estimates[i] = estimate;
    }

    return estimate;
}

template <typename TF, typename TPDF, typename TINVERSECDF>
double ImportanceSampledMonteCarloLDS(const TF& F, const TPDF& PDF, const TINVERSECDF& InverseCDF, std::vector<double>& estimates)
{
    estimates.resize(c_numSamples);
    double lds = 0.5;
    double estimate = 0.0f;
    for (size_t i = 0; i < c_numSamples; ++i)
    {
        double x = InverseCDF(lds);
        double y = F(x);
        double pdf = PDF(x);
        double value = y / pdf;
        AddSampleToRunningAverage(estimate, value, i);
        estimates[i] = estimate;
        lds = fmod(lds + c_goldeRatioConjugate, 1.0f);
    }

    return estimate;
}

template <typename TF, typename TPDF, typename TINVERSECDF>
double ImportanceSampledMonteCarloBlue(const TF& F, const TPDF& PDF, const TINVERSECDF& InverseCDF, std::vector<double>& estimates)
{
    std::vector<double> blueNoise;
    MakeBlueNoise(blueNoise, c_numSamples);

    estimates.resize(c_numSamples);
    double estimate = 0.0f;
    for (size_t i = 0; i < c_numSamples; ++i)
    {
        double x = InverseCDF(blueNoise[i]);
        double y = F(x);
        double pdf = PDF(x);
        double value = y / pdf;
        AddSampleToRunningAverage(estimate, value, i);
        estimates[i] = estimate;
    }

    return estimate;
}

template <typename TF, typename TPDF1, typename TINVERSECDF1, typename TPDF2, typename TINVERSECDF2>
double MultipleImportanceSampledMonteCarlo(const TF& F, const TPDF1& PDF1, const TINVERSECDF1& InverseCDF1, const TPDF2& PDF2, const TINVERSECDF2& InverseCDF2, std::vector<double>& estimates)
{
    std::mt19937 rng = GetRNG();
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    estimates.resize(c_numSamples);

    double estimate = 0.0f;
    for (size_t i = 0; i < c_numSamples; ++i)
    {
        double x1 = InverseCDF1(dist(rng));
        double y1 = F(x1);
        double pdf1 = PDF1(x1);
        double value1 = y1 / pdf1;

        double x2 = InverseCDF2(dist(rng));
        double y2 = F(x2);
        double pdf2 = PDF2(x2);
        double value2 = y2 / pdf2;

        double weight1 = pdf1 / (pdf1 + pdf2);
        double weight2 = pdf2 / (pdf1 + pdf2);

        double value = value1 * weight1 + value2 * weight2;

        AddSampleToRunningAverage(estimate, value, i);
        estimates[i] = estimate;
    }

    return estimate;
}

int main(int argc, char** argv)
{
    // y = sin(x)*sin(x) from 0 to pi
    {
        // The function we are integrating
        auto F = [](double x) -> double
        {
            return sin(x) * sin(x);
        };

        // the PDF and inverse CDF of a distribution we are using for integration
        auto PDF = [](double x) -> double
        {
            return sin(x) / 2.0f;
        };

        auto InverseCDF = [](double x) -> double
        {
            return 2.0 * asin(sqrt(x));
        };

        double c_actual = c_pi / 2.0f;

        // numerical integration
        std::vector<double> mcEstimates;
        double mc = MonteCarlo(F, mcEstimates);

        std::vector<double> mcblueEstimates;
        double mcblue = MonteCarloBlue(F, mcblueEstimates);

        std::vector<double> mcldsEstimates;
        double mclds = MonteCarloLDS(F, mcldsEstimates);

        std::vector<double> ismcEstimates;
        double ismc = ImportanceSampledMonteCarlo(F, PDF, InverseCDF, ismcEstimates);

        std::vector<double> ismcblueEstimates;
        double ismcblue = ImportanceSampledMonteCarloBlue(F, PDF, InverseCDF, ismcblueEstimates);

        std::vector<double> ismcldsEstimates;
        double ismclds = ImportanceSampledMonteCarloLDS(F, PDF, InverseCDF, ismcldsEstimates);

        // report results
        {
            // summary to screen
            printf("y = sin(x)*sin(x) from 0 to pi\n");
            printf("  mc       = %f  (%f)\n", mc, abs(mc - c_actual));
            printf("  mcblue   = %f  (%f)\n", mcblue, abs(mcblue - c_actual));
            printf("  mclds    = %f  (%f)\n", mclds, abs(mclds - c_actual));
            printf("  ismc     = %f  (%f)\n", ismc, abs(ismc - c_actual));
            printf("  ismcblue = %f  (%f)\n", ismcblue, abs(ismcblue - c_actual));
            printf("  ismclds  = %f  (%f)\n", ismclds, abs(ismclds - c_actual));
            printf("\n");

            // details to csv
            FILE* file = nullptr;
            fopen_s(&file, "out1.csv", "wb");
            fprintf(file, "\"index\",\"mc\",\"mcblue\",\"mclds\",\"ismc\",\"ismcblue\",\"ismclds\"\n");
            for (size_t i = 0; i < c_numSamples; ++i)
            {
                fprintf(file, "\"%zu\",", i);
                fprintf(file, "\"%f\",", max(abs(mcEstimates[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(mcblueEstimates[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(mcldsEstimates[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(ismcEstimates[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(ismcblueEstimates[i] - c_actual), c_minError));
                fprintf(file, "\"%f\"\n", max(abs(ismcldsEstimates[i] - c_actual), c_minError));
            }
            fclose(file);
        }
    }

    // y=sin(x)*2x from 0 to pi
    {
        // The function we are integrating
        auto F = [](double x) -> double
        {
            return sin(x) * 2.0f * x;
        };

        // the PDF and inverse CDF of distributions we are using for integration
        auto PDF1 = [](double x) -> double
        {
            // normalizing y=sin(x) from 0 to pi to integrate to 1
            return sin(x) / 2.0f;
        };

        auto InverseCDF1 = [](double x) -> double
        {
            // turning the PDF into a CDF, flipping x and y, and solving for y again
            return 2.0 * asin(sqrt(x));
        };

        auto PDF2 = [](double x) -> double
        {
            // normalizing y=2x from 0 to pi to integrate to 1
            return x * 2.0f / (c_pi * c_pi);
        };

        auto InverseCDF2 = [](double x) -> double
        {
            // turning the PDF into a CDF, flipping x and y, and solving for y again
            return c_pi * sqrt(x);
        };

        double c_actual = 2.0f * c_pi;

        // numerical integration
        std::vector<double> mcEstimates;
        double mc = MonteCarlo(F, mcEstimates);

        std::vector<double> mcblueEstimates;
        double mcblue = MonteCarloBlue(F, mcblueEstimates);

        std::vector<double> mcldsEstimates;
        double mclds = MonteCarloLDS(F, mcldsEstimates);

        std::vector<double> ismc1Estimates;
        double ismc1 = ImportanceSampledMonteCarlo(F, PDF1, InverseCDF1, ismc1Estimates);

        std::vector<double> ismcblue1Estimates;
        double ismcblue1 = ImportanceSampledMonteCarloBlue(F, PDF1, InverseCDF1, ismcblue1Estimates);

        std::vector<double> ismclds1Estimates;
        double ismclds1 = ImportanceSampledMonteCarloLDS(F, PDF1, InverseCDF1, ismclds1Estimates);

        std::vector<double> ismc2Estimates;
        double ismc2 = ImportanceSampledMonteCarlo(F, PDF2, InverseCDF2, ismc2Estimates);

        std::vector<double> ismcblue2Estimates;
        double ismcblue2 = ImportanceSampledMonteCarloBlue(F, PDF2, InverseCDF2, ismcblue2Estimates);

        std::vector<double> ismclds2Estimates;
        double ismclds2 = ImportanceSampledMonteCarloLDS(F, PDF2, InverseCDF2, ismclds2Estimates);

        std::vector<double> mismcEstimates;
        double mismc = MultipleImportanceSampledMonteCarlo(F, PDF1, InverseCDF1, PDF2, InverseCDF2, mismcEstimates);

        // report results
        {
            // summary to screen
            printf("y=sin(x)*2x from 0 to pi\n");
            printf("  mc        = %f  (%f)\n", mc, abs(mc - c_actual));
            printf("  mcblue    = %f  (%f)\n", mcblue, abs(mcblue - c_actual));
            printf("  mclds     = %f  (%f)\n", mclds, abs(mclds - c_actual));
            printf("  ismc1     = %f  (%f)\n", ismc1, abs(ismc1 - c_actual));
            printf("  ismcblue1 = %f  (%f)\n", ismcblue1, abs(ismcblue1 - c_actual));
            printf("  ismclds1  = %f  (%f)\n", ismclds1, abs(ismclds1 - c_actual));
            printf("  ismc2     = %f  (%f)\n", ismc2, abs(ismc2 - c_actual));
            printf("  ismcblue2 = %f  (%f)\n", ismcblue2, abs(ismcblue2 - c_actual));
            printf("  ismclds2  = %f  (%f)\n", ismclds2, abs(ismclds2 - c_actual));
            printf("  mismc     = %f  (%f)\n", mismc, abs(mismc - c_actual));
            printf("\n");

            // details to csv
            FILE* file = nullptr;
            fopen_s(&file, "out2.csv", "wb");
            fprintf(file, "\"index\",\"mc\",\"mcblue\",\"mclds\",\"ismc1\",\"ismcblue1\",\"ismclds1\",\"ismc2\",\"ismcblue2\",\"ismclds2\",\"mismc\"\n");
            for (size_t i = 0; i < c_numSamples; ++i)
            {
                fprintf(file, "\"%zu\",", i);
                fprintf(file, "\"%f\",", max(abs(mcEstimates[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(mcblueEstimates[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(mcldsEstimates[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(ismc1Estimates[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(ismcblue1Estimates[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(ismclds1Estimates[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(ismc2Estimates[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(ismcblue2Estimates[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(ismclds2Estimates[i] - c_actual), c_minError));
                fprintf(file, "\"%f\"\n", max(abs(mismcEstimates[i] - c_actual), c_minError));
            }
            fclose(file);
        }
    }

    // TODO: MIS in 2nd test doesn't seem to be working correctly! more error than IS's individually.
    // TODO: should 2nd test have LDS and Blue noise versions? maybe so... could be interesting!
    // TODO: do a 3rd test where it's a second test multiplied by some boolean function (visibility)

    return 0;
}

/*

TODO:
* sample 2 functions multiplied together - compare to sampling one, or the other, or uniform random
* maybe do more than 2 sampling strategies to see if you can do better?
* stochastically choose the technique? also with LDS (golden ratio)?
* maybe do a couple functions with the above?
* show error on log/log: scatter plot in open office, then format x/y axis to log


Blog:
* real simple / light on math / plainly described.
 * the other articles do a good job, but this is the "lemme just see the code" type of a post
* a follow up to 1d monte carlo / importance sampling:
 * https://blog.demofox.org/2018/06/12/monte-carlo-integration-explanation-in-1d/
 * note that you can give a different number of samples for each technique, if you think they will do better than others.
 ! could learn on the fly maybe. is that adaptive MIS?
 ! show how LDS + IS doesn't work well together. (actually, probably works fine in 1d). show blue noise

Links:
https://64.github.io/multiple-importance-sampling/
https://www.breakin.se/mc-intro/index.html#toc3
https://graphics.stanford.edu/courses/cs348b-03/papers/veach-chapter9.pdf

*/