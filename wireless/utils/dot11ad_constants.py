import math

###################
# 802.11ad - 2012 #
###################

# Table 8-0a -- Maximum MSDU and A-MSDU sizes
maxMsduSize = 7920 * 8  # [b]
maxAmsduSize = 7935 * 8  # [b]

# Table 21-4 -- Timing-related parameters
Fs = 2640e6  # OFDM sample rate [Hz]
Fc = Fs * 2 / 3  # SC chip rate [Hz]
Tc = 1 / Fc  # SC chip time [s]

Tseq = 128 * Tc  # [s]
T_STF = 17 * Tseq  # detection sequence duration [s]
T_CE = 9 * Tseq  # channel estimation sequence duration [s]
T_HEADER = 2 * 512 * Tc  # header duration (SC PHY) [s]

F_CCP = 1760e6  # control PHY chip rate [Hz]
T_CCP = 1 / F_CCP  # control PHY chip time [s]
T_STF_CP = 50 * Tseq  # control PHY short training field duration [s]
T_CE_CP = 9 * Tseq  # control PHY channel estimation field duration [s]


def get_T_Data(length, mcs):
    """
    Get the parameter T_{Data} as described in Table 21-4 of the 802.11ad-2012 standard.

    It refers to the time it takes to send the payload with the given MCS.

    Parameters
    ----------
    length : int
        The length of the payload [octets]
    mcs : int
        The MCS used. Only Control and SC PHY from the 802.11ad-2012 are supported (0 <= mcs <= 12)

    Returns
    -------
    float : the time it takes to send the payload [s]
    """
    N_BLKS = get_N_BLKS(length, mcs)
    T_Data = (N_BLKS * 512 + 64) * Tc
    return T_Data


# Table 21-20 -- Values of N_CBPB
N_CBPB = {'BPSK': 448,
          'QPSK': 896,
          '16QAM': 1792}


# Table 21-10 -- Modulation and coding scheme for the control PHY
# Table 21-23 -- Zero filling for SC BRP packets
def get_mcs_params(mcs):
    """
    Get all the parameters for the given MCS.

    Parameters
    ----------
    mcs : int
        The MCS used. Only Control and SC PHY from the 802.11ad-2012 are supported (0 <= mcs <= 12)

    Returns
    -------
    dict : mcs_params
        'modulation' : str
            ['BPSK', 'QPSK', '16QAM']
        'N_CBPB' : int
            Number of coded bits per symbol block
        'N_CBPS' : int
            Number of coded bits per symbol
        'rho' : int
            Repetitions
        'R' : float
            Code rate
        'data_rate' : float
            Nominal PHY data rate
    """
    assert 0 <= mcs <= 12, f"Only Control and SC PHY are supported. MCS{mcs} is not a valid MCS."
    mcs_params = {}

    # Modulation
    if mcs == 0:
        mcs_params['modulation'] = 'BPSK'
    elif 1 <= mcs <= 5:
        mcs_params['modulation'] = 'BPSK'
    elif 6 <= mcs <= 9:
        mcs_params['modulation'] = 'QPSK'
    elif 10 <= mcs <= 12:
        mcs_params['modulation'] = '16QAM'

    # N_CBPB
    mcs_params['N_CBPB'] = N_CBPB[mcs_params['modulation']]

    # N_CBPS
    if 1 <= mcs <= 5:
        mcs_params['N_CBPS'] = 1
    elif 6 <= mcs <= 9:
        mcs_params['N_CBPS'] = 2
    elif 10 <= mcs <= 12:
        mcs_params['N_CBPS'] = 4

    # Repetition
    if mcs == 0:
        mcs_params['rho'] = 1  # inferred, could not find
    elif mcs == 1:
        mcs_params['rho'] = 2
    elif 2 <= mcs <= 12:
        mcs_params['rho'] = 1

    # Code rate
    if mcs == 0:
        mcs_params['R'] = 1 / 2
    elif any(mcs == x for x in [1, 2, 6, 10]):
        mcs_params['R'] = 1 / 2
    elif any(mcs == x for x in [3, 7, 11]):
        mcs_params['R'] = 5 / 8
    elif any(mcs == x for x in [4, 8, 12]):
        mcs_params['R'] = 3 / 4
    elif any(mcs == x for x in [5, 9]):
        mcs_params['R'] = 13 / 16

    # Data rate
    # TODO remove from misc
    if mcs == 0:
        mcs_params['data_rate'] = 27.5e6
    elif mcs == 1:
        mcs_params['data_rate'] = 385e6
    elif mcs == 2:
        mcs_params['data_rate'] = 770e6
    elif mcs == 3:
        mcs_params['data_rate'] = 962.5e6
    elif mcs == 4:
        mcs_params['data_rate'] = 1155e6
    elif mcs == 5:
        mcs_params['data_rate'] = 1251.25e6
    elif mcs == 6:
        mcs_params['data_rate'] = 1540e6
    elif mcs == 7:
        mcs_params['data_rate'] = 1925e6
    elif mcs == 8:
        mcs_params['data_rate'] = 2310e6
    elif mcs == 9:
        mcs_params['data_rate'] = 2502.5e6
    elif mcs == 10:
        mcs_params['data_rate'] = 3080e6
    elif mcs == 11:
        mcs_params['data_rate'] = 3850e6
    elif mcs == 12:
        mcs_params['data_rate'] = 4620e6

    return mcs_params


# Table 21-31 -- DMG PHY characteristics
aSIFSTime = 3e-6  # [s]
aDataPreambleLength = 1891e-9  # [s]
aControlPHYPreambleLength = 4291e-9  # [s]
aSlotTime = 5e-6  # [s]
aCWmin = 15
aCWmax = 1023
aBRPminSCblocks = 18
aBRPTRNBlock = 4992
aSCGILength = 64
aSCBlockSize = 512
aPSDUMaxLength = 262143 * 8  # 256 kB - 1 B [b]

# 21.6.3.2.3.3 LDPC encoding process
L_CW = 672  # LDPC codeword length


def get_N_BLKS(length, mcs):
    """
    Get the number of SC PHY symbol blocks for the given length and MCS

    Parameters
    ----------
    length : int
        The length of the payload [octets]
    mcs : int
        The MCS used. Only Control and SC PHY from the 802.11ad-2012 are supported (0 <= mcs <= 12)

    Returns
    -------
    int : the number of SC PHY symbol blocks
    """
    mcs_params = get_mcs_params(mcs)
    N_CW = get_N_CW(length, mcs)
    N_BLKS = math.ceil((N_CW * L_CW) / mcs_params['N_CBPB'])
    return N_BLKS


def get_N_CW(length, mcs):
    """
    Get the number of LDPC codewords for SC PHY.

    Parameters
    ----------
    length : int
        The length of the payload [octets]
    mcs : int
        The MCS used. Only Control and SC PHY from the 802.11ad-2012 are supported (0 <= mcs <= 12)

    Returns
    -------
    int : the number of LDPC codewords
    """
    mcs_params = get_mcs_params(mcs)
    N_CW = math.ceil((length * 8) / (L_CW / mcs_params['rho'] * mcs_params['R']))
    return N_CW


# 21.12.3 TXTIME calculation
def get_TXTIME(length, mcs, N_TRN=0):
    """
    Get the value for the TXTIME parameter returned by the PLME-TXTIME.confirm primitive.

    It is equal to the total transmission time of a packet, including preamble, header, data, and TRN field.

    Parameters
    ----------
    length : int
        The length of the payload [octets]
    mcs : int
        The MCS used. Only Control and SC PHY from the 802.11ad-2012 are supported (0 <= mcs <= 12)
    N_TRN : int
        The training length field

    Returns
    -------
    float : the transmission time for a packet [s]
    """
    assert 0 <= mcs <= 12, f"Only Control and SC PHY are supported. MCS{mcs} is not a valid MCS."

    T_TRN_Unit = aBRPTRNBlock * Tc

    if mcs == 0:
        n_cw = get_N_CW(length, mcs)
        t_header_data = (11 * 8 + (length - 6) * 8 + n_cw * 168) * Tc * 32
        return T_STF_CP + T_CE_CP + t_header_data + N_TRN * T_TRN_Unit

    else:
        t_data = get_T_Data(length, mcs)
        if N_TRN == 0:
            return T_STF + T_CE + T_HEADER + t_data
        else:
            t_min = (aBRPminSCblocks * aSCBlockSize + aSCGILength) * Tc
            return T_STF + T_CE + T_HEADER + max(t_data, t_min) + N_TRN * T_TRN_Unit


###############
# 802.11-2016 #
###############

# 9.3.1 Control frames
# 9.3.1.2 RTS frame format
rts_length = (2 + 2 + 6 + 6 + 4) * 8  # Frame control + Duration + RA + TA + FCS

# 9.3.1.4 Ack frame format
ack_length = (2 + 2 + 6 + 4) * 8  # Frame control + Duration + RA + FCS

# 9.3.1.9 BlockAck frame format
ba_info_length = 2 + 10  # 9.3.1.9.3 Compressed BlockAck variant: Block Ack Starting Sequence Control + Block Ack Bitmap
blockack_length = (2 + 2 + 6 + 6 + 2 + ba_info_length + 4) * 8  # Frame control + Duration + RA + TA + BA Control +
# BA Information + FCS

# 9.3.1.14 DMG CTS frame format
dmg_cts_length = (2 + 2 + 6 + 6 + 4) * 8  # Frame control + Duration + RA + TA + FCS

# 10.3.7 DCF timing relations
DIFS = aSIFSTime + 2 * aSlotTime


# Figure 10-5 -- RTS/CTS/data/Ack and NAV setting
def get_total_tx_time(length_b, mcs, do_rts_cts=False):
    """
    Get the total transmission time for a packet.

    It also included the BlockACK, the Inter-Frame Spaces (IFSs), and, if required, the RTS/DMG CTS.

    Parameters
    ----------
    length_b : int
        The length of the payload [b]
    mcs : int
        The MCS used. Only Control and SC PHY from the 802.11ad-2012 are supported (0 <= mcs <= 12).
    do_rts_cts : bool
        Flag to include the RTS/DMG CTS if True.

    Returns
    -------
    float : the total transmission time for a packet [s]
    """
    # PSDU length is assumed to be in octets by the standard, but it is passed in bits from the simulator
    length_o = math.ceil(length_b / 8)

    if do_rts_cts:
        t_rts = get_TXTIME(rts_length, mcs=0)
        t_dmg_cts = get_TXTIME(dmg_cts_length, mcs=0)
        t_rts_cts = t_rts + aSIFSTime + t_dmg_cts + aSIFSTime
    else:
        t_rts_cts = 0

    t_data = get_TXTIME(length_o, mcs)
    t_ack = get_TXTIME(blockack_length, mcs=0)

    return t_rts_cts + t_data + aSIFSTime + t_ack + DIFS
