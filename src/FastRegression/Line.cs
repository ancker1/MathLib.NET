namespace FastRegression;

public static class Line
{
    public static (double a, double b) Fit(double[] x, double[] y)
    {
        if (x.Length != y.Length)
        {
            throw new ArgumentException($"The two input vectors must have the same length. The input vector {nameof(x)} has a length of {x.Length}, the input vector {nameof(y)} has a length of {y.Length}.");
        }

        if (x.Length <= 1)
        {
            throw new ArgumentException(
                $"To estimate a linear model at least two samples are required. The input vectors have a length of {x.Length}.");
        }

        double mean_x = 0d, mean_y = 0d;

        for (int i = 0; i < x.Length; i++)
        {
            mean_x += x[i];
            mean_y += y[i];
        }

        mean_x /= x.Length;
        mean_y /= y.Length;

        double covariance_xy = 0d, variance_x = 0d;
        for (int i = 0; i < x.Length; i++)
        {
            // Not the actual sample (co)variance, since it should be scaled with 1 / (N - 1). However, both the covariance and sample variance are scaled by the same factor so it can be omitted.
            double diff_x = x[i] - mean_x;
            covariance_xy += diff_x * (y[i] - mean_y);
            variance_x += diff_x * diff_x;
        }

        var b = covariance_xy / variance_x;
        return (mean_y - b * mean_x, b);
    }

    public static double PearsonCorrelation(double[] x, double[] y)
    {
        if (x.Length != y.Length)
        {
            throw new ArgumentException($"The two input vectors must have the same length. The input vector {nameof(x)} has a length of {x.Length}, the input vector {nameof(y)} has a length of {y.Length}.");
        }

        double mean_x = 0d, mean_y = 0d;

        for (int i = 0; i < x.Length; i++)
        {
            mean_x += x[i];
            mean_y += y[i];
        }

        mean_x /= x.Length;
        mean_y /= y.Length;

        double covariance_xy = 0d, variance_x = 0d, SST = 0d;
        for (int i = 0; i < x.Length; i++)
        {
            double diff_x = x[i] - mean_x;
            double diff_y = y[i] - mean_y;
            covariance_xy += diff_x * diff_y;
            variance_x += diff_x * diff_x;
            SST += diff_y * diff_y;
        }

        return covariance_xy / (variance_x * SST);
    }

    public static double Variance(double[] x)
    {
        if (x.Length <= 1)
        {
            throw new ArgumentException($"The number of samples must be greater than one in order to calculate the sample variance. The provided vector has a size of {x.Length}.");
        }

        double mean_x = 0d;

        for (int i = 0; i < x.Length; i++)
        {
            mean_x += x[i];
        }

        mean_x /= x.Length;

        double variance = 0d;

        for (int i = 0; i < x.Length; i++)
        {
            variance += (x[i] - mean_x) * (x[i] - mean_x);
        }

        return variance / (x.Length - 1);
    }

    public static double Covariance(double[] x, double[] y)
    {
        if (x.Length != y.Length)
        {
            throw new ArgumentException($"The two input vectors must have the same length. The input vector {nameof(x)} has a length of {x.Length}, the input vector {nameof(y)} has a length of {y.Length}.");
        }

        if (x.Length <= 1)
        {
            throw new ArgumentException($"The number of samples must be greater than one in order to calculate the sample variance. The provided vector has a size of {x.Length}.");
        }

        double mean_x = 0d, mean_y = 0d;

        for (int i = 0; i < x.Length; i++)
        {
            mean_x += x[i];
            mean_y += y[i];
        }

        mean_x /= x.Length;
        mean_y /= y.Length;

        double covariance_xy = 0d;
        for (int i = 0; i < x.Length; i++)
        {
            covariance_xy += (x[i] - mean_x) * (y[i] - mean_y);
        }

        return covariance_xy / (x.Length - 1);
    }

    public static double SlopeStandardError(double[] x, double[] y, double a, double b)
    {
        if (x.Length != y.Length)
        {
            throw new ArgumentException($"The two input vectors must have the same length. The input vector {nameof(x)} has a length of {x.Length}, the input vector {nameof(y)} has a length of {y.Length}.");
        }

        double mean_x = x.Sum();
        mean_x /= x.Length;

        double part_x = 0, SSE = 0;
        for (int i = 0; i < x.Length; i++)
        {
            part_x += (x[i] - mean_x) * (x[i] - mean_x);
            var yhat = (a * x[i] + b);
            SSE += (y[i] - yhat) * (y[i] - yhat);
        }

        return Math.Sqrt(1d / (x.Length - 2d) * SSE / part_x);
    }

    /// <summary>
    /// Assumes that the intercept is zero.
    /// </summary>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <param name="a"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentException"></exception>
    public static double SlopeStandardError(double[] x, double[] y, double a)
    {
        if (x.Length != y.Length)
        {
            throw new ArgumentException($"The two input vectors must have the same length. The input vector {nameof(x)} has a length of {x.Length}, the input vector {nameof(y)} has a length of {y.Length}.");
        }

        double SSE = 0d, SSX = 0d;
        for (int i = 0; i < x.Length; i++)
        {
            var yhat = (a * x[i]);
            SSE += (y[i] - yhat) * (y[i] - yhat);
            SSX += x[i] * x[i];
        }

        return Math.Sqrt(1d / (x.Length - 1) * SSE / SSX);
    }

    public static double InterceptStandardError(double[] x, double[] y, double a, double b)
    {
        if (x.Length != y.Length)
        {
            throw new ArgumentException($"The two input vectors must have the same length. The input vector {nameof(x)} has a length of {x.Length}, the input vector {nameof(y)} has a length of {y.Length}.");
        }

        double mx = 0d;
        for (int i = 0; i < x.Length; i++)
        {
            mx += x[i];
        }

        mx /= x.Length;

        double SSE = 0d, SSX = 0d, SSDiffX = 0d;
        for (int i = 0; i < x.Length; i++)
        {
            var yhat = a * x[i] + b;
            SSE += (y[i] - yhat) * (y[i] - yhat);
            SSX += x[i] * x[i];
            SSDiffX += (x[i] - mx) * (x[i] - mx);
        }

        return Math.Sqrt(1d / (x.Length * (x.Length - 2)) * SSE * SSX / SSDiffX);
    }

    // TODO: slope & intercept confidence intervals
}

