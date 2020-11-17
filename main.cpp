#include <random>
#include <vector>

#define DETERMINISTIC() false

static const size_t c_numSamples = 10000;

static const double c_pi = 3.14159265359;

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

int main(int argc, char** argv)
{
    {
        auto F = [](double x) -> double
        {
            return sin(x) * sin(x);
        };

        auto PDF = [](double x) -> double
        {
            return sin(x) / 2.0f;
        };

        auto InverseCDF = [](double x) -> double
        {
            return 2.0 * asin(sqrt(x));
        };

        double c_actual = c_pi / 2.0f;

        std::vector<double> mcEstimates;
        double mc = MonteCarlo(F, mcEstimates);

        std::vector<double> ismcEstimates;
        double ismc = ImportanceSampledMonteCarlo(F, PDF, InverseCDF, ismcEstimates);

        // report results
        {
            printf("mc   = %f  (%f)\nismc = %f  (%f)\n", mc, mc - c_actual, ismc, ismc - c_actual);

            FILE* file = nullptr;
            fopen_s(&file, "out.csv", "wb");
            fprintf(file, "\"index\",\"mc\",\"ismc\"\n");
            for (size_t i = 0; i < c_numSamples; ++i)
                fprintf(file, "\"%zu\",\"%f\",\"%f\"\n", i, abs(mcEstimates[i] - c_actual), abs(ismcEstimates[i] - c_actual));
            fclose(file);
        }
    }
    return 0;
}

/*

TODO:
* sample 2 functions multiplied together - compare to sampling one, or the other, or uniform random
* maybe do more than 3 sampling strategies to see if you can do better?
* stochastically choose the technique? with LDS, like golden ratio?
* could also LDS sample as a comparison
* show the results
* maybe do a couple functions with the above?
* show error on log/log: scatter plot in open office, then format x/y axis to log

Blog:
* real simple / light on math / plainly described.
 * the other articles do a good job, but this is the "lemme just see the code" type of a post
* a follow up to 1d monte carlo / importance sampling:
 * https://blog.demofox.org/2018/06/12/monte-carlo-integration-explanation-in-1d/
 * note that you can give a different number of samples for each technique, if you think they will do better than others.
 ! could learn on the fly maybe. is that adaptive MIS?

Links:
https://64.github.io/multiple-importance-sampling/
https://www.breakin.se/mc-intro/index.html#toc3

*/