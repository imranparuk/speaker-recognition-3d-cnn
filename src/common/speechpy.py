from __future__ import division

import numpy as np
import math

import numpy as np
from scipy.fftpack import dct

def stack_frames(
        sig,
        sampling_frequency,
        frame_length=0.020,
        frame_stride=0.020,
        filter=lambda x: np.ones((x,)),
        zero_padding=False):
    """Frame a signal into overlapping frames.
    Args:
        sig (array): The audio signal to frame of size (N,).
        sampling_frequency (int): The sampling frequency of the signal.
        frame_length (float): The length of the frame in second.
        frame_stride (float): The stride between frames.
        filter (array): The time-domain filter for applying to each frame.
            By default it is one so nothing will be changed.
        zero_padding (bool): If the samples is not a multiple of
            frame_length(number of frames sample), zero padding will
            be done for generating last frame.
    Returns:
            array: Stacked_frames-Array of frames of size (number_of_frames x frame_len).
    """

    # Check dimension
    s = "Signal dimention should be of the format of (N,) but it is %s instead"
    #print(sig)
    assert sig.ndim == 1, s % str(sig.shape)

    # Initial necessary values
    length_signal = sig.shape[0]
    # print(length_signal)

    frame_sample_length = int(
        np.round(
            sampling_frequency *
            frame_length))  # Defined by the number of samples
    frame_stride = float(np.round(sampling_frequency * frame_stride))

    # print("frame_sample_length: ", frame_sample_length)
    # print("frame_stride: ", frame_stride)

    # Zero padding is done for allocating space for the last frame.
    if zero_padding:
        # Calculation of number of frames
        numframes = (int(math.ceil((length_signal
                                      - frame_sample_length) / frame_stride)))


        #print(numframes,length_signal,frame_sample_length,frame_stride)

        # Zero padding
        len_sig = int(numframes * frame_stride + frame_sample_length)
        additive_zeros = np.zeros((len_sig - length_signal,))
        signal = np.concatenate((sig, additive_zeros))

    else:
        # No zero padding! The last frame which does not have enough
        # samples(remaining samples <= frame_sample_length), will be dropped!
        numframes = int(math.floor((length_signal
                          - frame_sample_length) / frame_stride))

        # new length
        len_sig = int((numframes - 1) * frame_stride + frame_sample_length)
        signal = sig[0:len_sig]
        # print("numframes: ", numframes)
        # print("length_signal: ", length_signal)
        # print("frame_sample_length: ", frame_sample_length)
        # print("frame_stride: ", frame_stride)
        # print("a: ", np.tile(np.arange(0, frame_sample_length), (numframes, 1)).shape)
        # print("b: ",  np.tile(np.arange(0, numframes * frame_stride, frame_stride), (frame_sample_length, 1)).T.shape)
    # (1, 361, 80)

    # exit()
    # Getting the indices of all frames.
    indices = np.tile(np.arange(0, frame_sample_length), (numframes, 1)) + np.tile(np.arange(0, numframes * frame_stride, frame_stride), (frame_sample_length, 1)).T

    # print("indices: ", indices)

    indices = np.array(indices, dtype=np.int32)

    # Extracting the frames based on the allocated indices.
    frames = signal[indices]

    # Apply the windows function
    window = np.tile(filter(frame_sample_length), (numframes, 1))
    Extracted_Frames = frames * window
    return Extracted_Frames

def frequency_to_mel(f):
    """converting from frequency to Mel scale.
    :param f: The frequency values(or a single frequency) in Hz.
    :returns: The mel scale values(or a single mel).
    """
    return 1127 * np.log(1 + f / 700.)


def mel_to_frequency(mel):
    """converting from Mel scale to frequency.
    :param mel: The mel scale values(or a single mel).
    :returns: The frequency values(or a single frequency) in Hz.
    """
    return 700 * (np.exp(mel / 1127.0) - 1)


def triangle(x, left, middle, right):
    out = np.zeros(x.shape)
    out[x <= left] = 0
    out[x >= right] = 0
    first_half = np.logical_and(left < x, x <= middle)
    out[first_half] = (x[first_half] - left) / (middle - left)
    second_half = np.logical_and(middle <= x, x < right)
    out[second_half] = (right - x[second_half]) / (right - middle)
    return out


def zero_handling(x):
    """
    This function handle the issue with zero values if the are exposed
    to become an argument for any log function.
    :param x: The vector.
    :return: The vector with zeros substituted with epsilon values.
    """
    return np.where(x == 0, np.finfo(float).eps, x)

def fft_spectrum(frames, fft_points=512):
    """This function computes the one-dimensional n-point discrete Fourier
    Transform (DFT) of a real-valued array by means of an efficient algorithm
    called the Fast Fourier Transform (FFT). Please refer to
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.rfft.html
    for further details.
    Args:
        frames (array): The frame array in which each row is a frame.
        fft_points (int): The length of FFT. If fft_length is greater than frame_len, the frames will be zero-padded.
    Returns:
            array: The fft spectrum.
            If frames is an num_frames x sample_per_frame matrix, output
            will be num_frames x FFT_LENGTH.
    """
    SPECTRUM_VECTOR = np.fft.rfft(frames, n=fft_points, axis=-1, norm=None)
    return np.absolute(SPECTRUM_VECTOR)

def power_spectrum(frames, fft_points=512):
    """Power spectrum of each frame.
    Args:
        frames (array): The frame array in which each row is a frame.
        fft_points (int): The length of FFT. If fft_length is greater than frame_len, the frames will be zero-padded.
    Returns:
            array: The power spectrum.
            If frames is an num_frames x sample_per_frame matrix, output
            will be num_frames x fft_length.
    """
    return 1.0 / fft_points * np.square(fft_spectrum(frames, fft_points))

def filterbanks(
        num_filter,
        coefficients,
        sampling_freq,
        low_freq=None,
        high_freq=None):
    """Compute the Mel-filterbanks. Each filter will be stored in one rows.
    The columns correspond to fft bins.
    Args:
        num_filter (int): the number of filters in the filterbank, default 20.
        coefficients (int): (fftpoints//2 + 1). Default is 257.
        sampling_freq (float): the samplerate of the signal we are working
            with. It affects mel spacing.
        low_freq (float): lowest band edge of mel filters, default 0 Hz
        high_freq (float): highest band edge of mel filters,
            default samplerate/2
    Returns:
           array: A numpy array of size num_filter x (fftpoints//2 + 1)
               which are filterbank
    """
    high_freq = high_freq or sampling_freq / 2
    low_freq = low_freq or 300
    s = "High frequency cannot be greater than half of the sampling frequency!"
    assert high_freq <= sampling_freq / 2, s
    assert low_freq >= 0, "low frequency cannot be less than zero!"

    # Computing the Mel filterbank
    # converting the upper and lower frequencies to Mels.
    # num_filter + 2 is because for num_filter filterbanks we need
    # num_filter+2 point.
    mels = np.linspace(
        frequency_to_mel(low_freq),
        frequency_to_mel(high_freq),
        num_filter + 2)

    # we should convert Mels back to Hertz because the start and end-points
    # should be at the desired frequencies.
    hertz = mel_to_frequency(mels)

    # The frequency resolution required to put filters at the
    # exact points calculated above should be extracted.
    #  So we should round those frequencies to the closest FFT bin.
    freq_index = (
        np.floor(
            (coefficients +
             1) *
            hertz /
            sampling_freq)).astype(int)

    # Initial definition
    filterbank = np.zeros([num_filter, coefficients])

    # The triangular function for each filter
    for i in range(0, num_filter):
        left = int(freq_index[i])
        middle = int(freq_index[i + 1])
        right = int(freq_index[i + 2])
        z = np.linspace(left, right, num=right - left + 1)
        filterbank[i,
                   left:right + 1] = triangle(z,
                                                        left=left,
                                                        middle=middle,
                                                        right=right)

    return filterbank

def mfe(signal, sampling_frequency, frame_length=0.020, frame_stride=0.01,
        num_filters=40, fft_length=512, low_frequency=0, high_frequency=None):
    """Compute Mel-filterbank energy features from an audio signal.

    Args:
         signal (array): the audio signal from which to compute features.
             Should be an N x 1 array
         sampling_frequency (int): the sampling frequency of the signal
             we are working with.
         frame_length (float): the length of each frame in seconds.
             Default is 0.020s
         frame_stride (float): the step between successive frames in seconds.
             Default is 0.02s (means no overlap)
         num_filters (int): the number of filters in the filterbank,
             default 40.
         fft_length (int): number of FFT points. Default is 512.
         low_frequency (float): lowest band edge of mel filters.
             In Hz, default is 0.
         high_frequency (float): highest band edge of mel filters.
             In Hz, default is samplerate/2
    Returns:
              array: features - the energy of fiterbank of size num_frames x num_filters. The energy of each frame: num_frames x 1
    """

    # Convert to float
    signal = signal.astype(float)

    # Stack frames
    frames = stack_frames(
        signal,
        sampling_frequency=sampling_frequency,
        frame_length=frame_length,
        frame_stride=frame_stride,
        filter=lambda x: np.ones(
            (x,
             )),
        zero_padding=False)

    # getting the high frequency
    high_frequency = high_frequency or sampling_frequency / 2

    # calculation of the power sprectum
    power_spectrums = power_spectrum(frames, fft_length)
    coefficients = power_spectrums.shape[1]
    # this stores the total energy in each frame
    frame_energies = np.sum(power_spectrums, 1)

    # Handling zero enegies.
    frame_energies = zero_handling(frame_energies)

    # Extracting the filterbank
    filter_banks = filterbanks(
        num_filters,
        coefficients,
        sampling_frequency,
        low_frequency,
        high_frequency)

    # Filterbank energies
    features = np.dot(power_spectrums, filter_banks.T)
    features = zero_handling(features)


    return features, frame_energies

def cmvn(vec, variance_normalization=False):
    """ This function is aimed to perform global cepstral mean and
        variance normalization (CMVN) on input feature vector "vec".
        The code assumes that there is one observation per row.
    Args:
        vec (array): input feature matrix
            (size:(num_observation,num_features))
        variance_normalization (bool): If the variance
            normilization should be performed or not.
    Return:
          array: The mean(or mean+variance) normalized feature vector.
    """
    eps = 2**-30
    rows, cols = vec.shape

    # Mean calculation
    norm = np.mean(vec, axis=0)
    norm_vec = np.tile(norm, (rows, 1))

    # Mean subtraction
    mean_subtracted = vec - norm_vec

    # Variance normalization
    if variance_normalization:
        stdev = np.std(mean_subtracted, axis=0)
        stdev_vec = np.tile(stdev, (rows, 1))
        output = mean_subtracted / (stdev_vec + eps)
    else:
        output = mean_subtracted

    return output

def mfcc(
        signal,
        sampling_frequency,
        frame_length=0.020,
        frame_stride=0.01,
        num_cepstral=13,
        num_filters=40,
        fft_length=512,
        low_frequency=0,
        high_frequency=None,
        dc_elimination=True):
    """Compute MFCC features from an audio signal.
    Args:
         signal (array): the audio signal from which to compute features.
             Should be an N x 1 array
         sampling_frequency (int): the sampling frequency of the signal
             we are working with.
         frame_length (float): the length of each frame in seconds.
             Default is 0.020s
         frame_stride (float): the step between successive frames in seconds.
             Default is 0.02s (means no overlap)
         num_filters (int): the number of filters in the filterbank,
             default 40.
         fft_length (int): number of FFT points. Default is 512.
         low_frequency (float): lowest band edge of mel filters.
             In Hz, default is 0.
         high_frequency (float): highest band edge of mel filters.
             In Hz, default is samplerate/2
         num_cepstral (int): Number of cepstral coefficients.
         dc_elimination (bool): hIf the first dc component should
             be eliminated or not.
    Returns:
        array: A numpy array of size (num_frames x num_cepstral) containing mfcc features.
    """
    feature, energy = mfe(signal, sampling_frequency=sampling_frequency,
                          frame_length=frame_length, frame_stride=frame_stride,
                          num_filters=num_filters, fft_length=fft_length,
                          low_frequency=low_frequency,
                          high_frequency=high_frequency)
    if len(feature) == 0:
        return np.empty((0, num_cepstral))
    feature = np.log(feature)
    feature = dct(feature, type=2, axis=-1, norm='ortho')[:, :num_cepstral]

    # replace first cepstral coefficient with log of frame energy for DC
    # elimination.
    if dc_elimination:
        feature[:, 0] = np.log(energy)
    return feature


def lmfe(signal, sampling_frequency, frame_length=0.020, frame_stride=0.01,
         num_filters=40, fft_length=512, low_frequency=0, high_frequency=None):
    """Compute log Mel-filterbank energy features from an audio signal.
    Args:
         signal (array): the audio signal from which to compute features.
             Should be an N x 1 array
         sampling_frequency (int): the sampling frequency of the signal
             we are working with.
         frame_length (float): the length of each frame in seconds.
             Default is 0.020s
         frame_stride (float): the step between successive frames in seconds.
             Default is 0.02s (means no overlap)
         num_filters (int): the number of filters in the filterbank,
             default 40.
         fft_length (int): number of FFT points. Default is 512.
         low_frequency (float): lowest band edge of mel filters.
             In Hz, default is 0.
         high_frequency (float): highest band edge of mel filters.
             In Hz, default is samplerate/2
    Returns:
              array: Features - The log energy of fiterbank of size num_frames x num_filters frame_log_energies. The log energy of each frame num_frames x 1
    """

    feature, frame_energies = mfe(signal,
                                  sampling_frequency=sampling_frequency,
                                  frame_length=frame_length,
                                  frame_stride=frame_stride,
                                  num_filters=num_filters,
                                  fft_length=fft_length,
                                  low_frequency=low_frequency,
                                  high_frequency=high_frequency)
    feature = np.log(feature)

    return feature
