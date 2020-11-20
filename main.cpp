#include <random>
#include <vector>

#define DETERMINISTIC() true
#define DO_BLUE_NOISE() false  // blue noise is very slow to generate, and doesn't have good convergence speed

static const size_t c_numSamples = 1000;
static const size_t c_numTests = 10000;

static const double c_pi = 3.14159265359;
static const double c_goldeRatioConjugate = 0.61803398875;
static const double c_sqrt2 = sqrt(2.0);
static const double c_minError = 0.00001; // to avoid errors when showing on a log plot

std::mt19937 GetRNG(int seed)
{
#if DETERMINISTIC()
    std::mt19937 mt(seed);
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

void AddSampleToRunningAverage(double &average, double newValue, size_t sampleCount)
{
    // Incremental averaging: lerp from old value to new value by 1/(sampleCount+1)
    // https://blog.demofox.org/2016/08/23/incremental-averaging/
    double t = 1.0 / double(sampleCount + 1);
    average = average * (1.0 - t) + newValue * t;
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

void IntegrateResult(Result& total, const Result& sample, int sampleCount)
{
    AddSampleToRunningAverage(total.estimate, sample.estimate, sampleCount);
    total.estimates.resize(sample.estimates.size());
    for (size_t index = 0; index < total.estimates.size(); ++index)
        AddSampleToRunningAverage(total.estimates[index], sample.estimates[index], sampleCount);
}

int main(int argc, char** argv)
{
    printf("%zu tests, with %zu samples each.\n\n", c_numTests, c_numSamples);

    // y = sin(x)*sin(x) from 0 to pi
    {
        struct Tests
        {
            Result mc;
            Result mcblue;
            Result mclds;
            Result ismc;
            Result ismcblue;
            Result ismclds;
        };
        Tests tests, testsTotal;

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
            MonteCarlo(F, tests.mc, testIndex);
            MonteCarloBlue(F, tests.mcblue, testIndex);
            MonteCarloLDS(F, tests.mclds, testIndex);
            ImportanceSampledMonteCarlo(F, PDF, InverseCDF, tests.ismc, testIndex);
            ImportanceSampledMonteCarloBlue(F, PDF, InverseCDF, tests.ismcblue, testIndex);
            ImportanceSampledMonteCarloLDS(F, PDF, InverseCDF, tests.ismclds, testIndex);

            IntegrateResult(testsTotal.mc, tests.mc, testIndex);
            IntegrateResult(testsTotal.mcblue, tests.mcblue, testIndex);
            IntegrateResult(testsTotal.mclds, tests.mclds, testIndex);
            IntegrateResult(testsTotal.ismc, tests.ismc, testIndex);
            IntegrateResult(testsTotal.ismcblue, tests.ismcblue, testIndex);
            IntegrateResult(testsTotal.ismclds, tests.ismclds, testIndex);
        }

        // report results
        {
            // summary to screen
            printf("y = sin(x)*sin(x) from 0 to pi\n");
            printf("  mc       = %f  (%f)\n", testsTotal.mc.estimate, abs(testsTotal.mc.estimate - c_actual));
            printf("  mcblue   = %f  (%f)\n", testsTotal.mcblue.estimate, abs(testsTotal.mcblue.estimate - c_actual));
            printf("  mclds    = %f  (%f)\n", testsTotal.mclds.estimate, abs(testsTotal.mclds.estimate - c_actual));
            printf("  ismc     = %f  (%f)\n", testsTotal.ismc.estimate, abs(testsTotal.ismc.estimate - c_actual));
            printf("  ismcblue = %f  (%f)\n", testsTotal.ismcblue.estimate, abs(testsTotal.ismcblue.estimate - c_actual));
            printf("  ismclds  = %f  (%f)\n", testsTotal.ismclds.estimate, abs(testsTotal.ismclds.estimate - c_actual));
            printf("\n");

            // details to csv
            FILE* file = nullptr;
            fopen_s(&file, "out1.csv", "wb");
            fprintf(file, "\"index\",\"mc\",\"mcblue\",\"mclds\",\"ismc\",\"ismcblue\",\"ismclds\"\n");
            for (size_t i = 0; i < c_numSamples; ++i)
            {
                fprintf(file, "\"%zu\",", i);
                fprintf(file, "\"%f\",", max(abs(testsTotal.mc.estimates[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(testsTotal.mcblue.estimates[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(testsTotal.mclds.estimates[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(testsTotal.ismc.estimates[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(testsTotal.ismcblue.estimates[i] - c_actual), c_minError));
                fprintf(file, "\"%f\"\n", max(abs(testsTotal.ismclds.estimates[i] - c_actual), c_minError));
            }
            fclose(file);
        }
    }

    // y=sin(x)*2x from 0 to pi
    {
        struct Tests
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
        };
        Tests tests, testsTotal;

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
            MonteCarlo(F, tests.mc, testIndex);
            MonteCarloBlue(F, tests.mcblue, testIndex);
            MonteCarloLDS(F, tests.mclds, testIndex);
            ImportanceSampledMonteCarlo(F, PDF1, InverseCDF1, tests.ismc1, testIndex);
            ImportanceSampledMonteCarloBlue(F, PDF1, InverseCDF1, tests.ismcblue1, testIndex);
            ImportanceSampledMonteCarloLDS(F, PDF1, InverseCDF1, tests.ismclds1, testIndex);
            ImportanceSampledMonteCarlo(F, PDF2, InverseCDF2, tests.ismc2, testIndex);
            ImportanceSampledMonteCarloBlue(F, PDF2, InverseCDF2, tests.ismcblue2, testIndex);
            ImportanceSampledMonteCarloLDS(F, PDF2, InverseCDF2, tests.ismclds2, testIndex);
            MultipleImportanceSampledMonteCarlo(F, PDF1, InverseCDF1, PDF2, InverseCDF2, tests.mismc, testIndex);
            MultipleImportanceSampledMonteCarloLDS(F, PDF1, InverseCDF1, PDF2, InverseCDF2, tests.mismclds, testIndex);

            IntegrateResult(testsTotal.mc, tests.mc, testIndex);
            IntegrateResult(testsTotal.mcblue, tests.mcblue, testIndex);
            IntegrateResult(testsTotal.mclds, tests.mclds, testIndex);
            IntegrateResult(testsTotal.ismc1, tests.ismc1, testIndex);
            IntegrateResult(testsTotal.ismcblue1, tests.ismcblue1, testIndex);
            IntegrateResult(testsTotal.ismclds1, tests.ismclds1, testIndex);
            IntegrateResult(testsTotal.ismc2, tests.ismc2, testIndex);
            IntegrateResult(testsTotal.ismcblue2, tests.ismcblue2, testIndex);
            IntegrateResult(testsTotal.ismclds2, tests.ismclds2, testIndex);
            IntegrateResult(testsTotal.mismc, tests.mismc, testIndex);
            IntegrateResult(testsTotal.mismclds, tests.mismclds, testIndex);
        }

        // report results
        {
            // summary to screen
            printf("y=sin(x)*2x from 0 to pi\n");
            printf("  mc        = %f  (%f)\n", testsTotal.mc.estimate, abs(testsTotal.mc.estimate - c_actual));
            printf("  mcblue    = %f  (%f)\n", testsTotal.mcblue.estimate, abs(testsTotal.mcblue.estimate - c_actual));
            printf("  mclds     = %f  (%f)\n", testsTotal.mclds.estimate, abs(testsTotal.mclds.estimate - c_actual));
            printf("  ismc1     = %f  (%f)\n", testsTotal.ismc1.estimate, abs(testsTotal.ismc1.estimate - c_actual));
            printf("  ismcblue1 = %f  (%f)\n", testsTotal.ismcblue1.estimate, abs(testsTotal.ismcblue1.estimate - c_actual));
            printf("  ismclds1  = %f  (%f)\n", testsTotal.ismclds1.estimate, abs(testsTotal.ismclds1.estimate - c_actual));
            printf("  ismc2     = %f  (%f)\n", testsTotal.ismc2.estimate, abs(testsTotal.ismc2.estimate - c_actual));
            printf("  ismcblue2 = %f  (%f)\n", testsTotal.ismcblue2.estimate, abs(testsTotal.ismcblue2.estimate - c_actual));
            printf("  ismclds2  = %f  (%f)\n", testsTotal.ismclds2.estimate, abs(testsTotal.ismclds2.estimate - c_actual));
            printf("  mismc     = %f  (%f)\n", testsTotal.mismc.estimate, abs(testsTotal.mismc.estimate - c_actual));
            printf("  mismclds  = %f  (%f)\n", testsTotal.mismclds.estimate, abs(testsTotal.mismclds.estimate - c_actual));
            printf("\n");

            // details to csv
            FILE* file = nullptr;
            fopen_s(&file, "out2.csv", "wb");
            fprintf(file, "\"index\",\"mc\",\"mcblue\",\"mclds\",\"ismc1\",\"ismcblue1\",\"ismclds1\",\"ismc2\",\"ismcblue2\",\"ismclds2\",\"mismc\",\"mismclds\"\n");
            for (size_t i = 0; i < c_numSamples; ++i)
            {
                fprintf(file, "\"%zu\",", i);
                fprintf(file, "\"%f\",", max(abs(testsTotal.mc.estimates[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(testsTotal.mcblue.estimates[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(testsTotal.mclds.estimates[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(testsTotal.ismc1.estimates[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(testsTotal.ismcblue1.estimates[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(testsTotal.ismclds1.estimates[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(testsTotal.ismc2.estimates[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(testsTotal.ismcblue2.estimates[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(testsTotal.ismclds2.estimates[i] - c_actual), c_minError));
                fprintf(file, "\"%f\",", max(abs(testsTotal.mismc.estimates[i] - c_actual), c_minError));
                fprintf(file, "\"%f\"\n", max(abs(testsTotal.mismclds.estimates[i] - c_actual), c_minError));
            }
            fclose(file);
        }
    }

    return 0;
}

/*

TODO:
* stochastically choose the technique? also with LDS (golden ratio)? yes do this.
* should report variance too


Blog:
* you divide f(x) by the sum of probabilities of the techniques for x. albegraicly this makes sense, but not sure intuitively.
* real simple / light on math / plainly described.
 * the other articles do a good job, but this is the "lemme just see the code" type of a post
* a follow up to 1d monte carlo / importance sampling:
 * https://blog.demofox.org/2018/06/12/monte-carlo-integration-explanation-in-1d/
 * note that you can give a different number of samples for each technique, if you think they will do better than others.
 ! could learn on the fly maybe. is that adaptive MIS?
 ! show how LDS + IS doesn't work well together. (actually, probably works fine in 1d). show blue noise
 * MIS takes more samples.
 * show error on log/log: scatter plot in open office, then format x/y axis to log
 * if no bias, reducing variance is the same as removing error.

 "it is allowable for some of the p_i to be specialized sampling techniques that concentrate on specific regions of the integrand"
 * aka if you have at least one technique for each section this will work. can have multiple techniques for multiple sections
 * a technique doesn't need to fit the whole function though!

Links:
https://64.github.io/multiple-importance-sampling/
https://www.breakin.se/mc-intro/index.html#toc3
https://graphics.stanford.edu/courses/cs348b-03/papers/veach-chapter9.pdf

*/