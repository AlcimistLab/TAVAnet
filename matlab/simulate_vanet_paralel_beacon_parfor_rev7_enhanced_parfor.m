% =========================================================================
% Vehicular Communication Simulation Script
% Using SUMO FCD Data and UBX-V2X Library for IEEE 802.11bd Experiment
% Made by:
% =========================================================================

% =========================================================================
% Parameter Initialization
% =========================================================================
% fcd-output-250m-45-vehicle-500second.xml
% General Simulation Parameters
simulationParams = struct( ...
    'fcdFilePath', 'fcd-output-250m-45-vehicle-500second.xml', ...
    'numVehicles', 45, ...
    'beaconRate', 10, ...              % Beacons per second
    'simulationDuration',100, ...     % Simulation time in seconds
    'cbrFormulaType', 1, ...           % Type of Channel Busy Ratio (CBR) formula (1, 2 or 3)
    'enableRL', true, ...              % Enable Reinforcement Learning (RL)
    'safetyMsgProbability', 0.00, ...   % Safety Messages Probability
    'fcdTimeLimit', 100, ...             % Time limit for load fcd 
    'trafficProfile', 'highway_los' ... % Traffic Profile
);

%Phy Layer Parameters
phyParams = struct( ...
    'transmissionPower', 20, ... % Transmission power in dBm
    'bandwidth', 20e6, ... % Bandwidth in Hz
    'noiseFigure', 5, ... % Noise figure in dB
    'channelModel', 4, ... % Channel model: 0: AWGN (Additive White Gaussian Noise), 1: R-LOS (Rural Line-of-Sight), 2: UA-LOS (Urban Area Line-of-Sight), 3: C-NLOS (Canyon Non-Line-of-Sight), 4: H-LOS (Highway Line-of-Sight), 5: H-NLOS (Highway Non-Line-of-Sight), 6: R-LOS-ENH (Enhanced Rural Line-of-Sight), 7: UA-LOS-ENH (Enhanced Urban Area Line-of-Sight), 8: C-NLOS-ENH (Enhanced Canyon Non-Line-of-Sight), 9: H-LOS-ENH (Enhanced Highway Line-of-Sight), 10: H-NLOS-ENH (Enhanced Highway Non-Line-of-Sight)  
    'c_d', 6, ... % Default Coded bits per OFDM symbol
    'MCS', 0, ... % Default MCS level
    'payloadLength', 350, ... % Payload of the PHY Layer in Bytes
    'G_t', 5, ... % Antenna Gain Transmitt
    'G_r', 5, ... % Antenna Gain Receive
    'frequency', 5.9e9 ... % Frequency in Hz
);

% MAC Layer Parameters
macParams = struct( ...
    'macProtocol', 'CSMA-CA', ...     % MAC access scheme used
    'useRTSCTS', false, ...           % VANET MAC disables RTS/CTS
    'ackEnabled', true, ...           % Use ACK for successful delivery confirmation
    'retryLimit', 3, ...              % Max MAC retransmissions (Stop-and-Wait ARQ)
    'cwMin', 15, ...                  % Minimum contention window (slots)
    'cwMax', 1023, ...                % Maximum contention window (slots)
    'slotTime', 13e-6, ...            % Slot time (9 µs typical for 802.11bd)
    'sifs', 16e-6, ...                % Short Interframe Space (32 µs for 802.11bd)
    'difs', 34e-6, ...                % DCF Interframe Space (58 µs for 802.11bd)
    'macHeaderBytes', 34, ...        % MAC header size in bytes (e.g. 802.11 format)
    'ackTimeout', 300e-6, ...        % Timeout waiting for ACK (300 µs)
    'addressing', 'logical' ...      % Logical MAC address mode (for mapping)
);

% CBR Calculation Parameters
cbrParams = struct( ...
    'S', 1e-10, ...         % Sensitivity threshold (W)
    'beta', 3, ...          % Path loss exponent
    'm', 1, ...             % Shape parameter
    'wavelength', 0.0508, ...% Carrier wavelength (5.9 GHz)
    'c_d', 6, ...           % Coded bits per OFDM symbol
    'b_st', 22, ...         % Service and tail bits
    'n', 536 * 8, ...       % Packet length in bits
    'bandwidth', 20e6, ...  % Bandwidth in Hz
    't_ps', 40e-6 ...       % Preamble/signal field duration (s)
);


% =========================================================================
% Load FCD Data and Resolve Duplicates
% =========================================================================
disp('[INFO] Loading and sanitizing FCD data...');
try
    % Load FCD data
    fcdData = load_fcd_data_type3(simulationParams.fcdFilePath, 0, simulationParams.fcdTimeLimit);
    % fcdData = load_fcd_data(simulationParams.fcdFilePath);
    % Resolve duplicate vehicle positions
    disp('[INFO] Resolving duplicate vehicle positions...');
    fcdData = resolve_duplicate_positions(fcdData);
    disp('[INFO] Duplicate positions resolved successfully.');
catch ME
    error('[ERROR] Failed to load or sanitize FCD data: %s', ME.message);
end

% Ensure vehicle IDs in FCD data are properly formatted
disp('[INFO] Standardizing vehicle IDs in FCD data...');
for i = 1:length(fcdData)
    fcdData(i).id = enforce_vehicle_format(fcdData(i).id);
end

% =========================================================================
% Calculate Pairwise Distances, Neighbor Status, and Save Results
% =========================================================================
%{
%disp('[INFO] Calculating pairwise distances and neighbors...');
defaultCommRange = 175; % Default communication range in meters

% Specify the output file for saving neighbor data
outputFile = 'neighbor_data.xlsx'; % Change to 'neighbor_data.csv' if preferred

% Calculate distances, neighbors, and save results to a file
disp('[INFO] Calculating neighbors, communication ranges, and received power...');
save_neighbor_data(fcdData, phyParams.commRange, outputFile, phyParams);
disp('[INFO] Neighbor data saved successfully.');
%}
% =========================================================================
% RL Initialization
% =========================================================================

% RL Socket Initialization
if simulationParams.enableRL
    disp('[INFO] Initializing RL socket connection...');
    rlClient = tcpclient('127.0.0.1', 5000); % Replace IP/port as required
end



% =========================================================================
% Initialize Performance Metrics
% =========================================================================
% Extract unique vehicle IDs from FCD data
uniqueVehicleIDs = unique({fcdData.id}); 

% Initialize vehicleMetrics for each vehicle
disp('[INFO] Initializing vehicle metrics...');
vehicleMetrics = struct();
for vIdx = 1:length(uniqueVehicleIDs)
    vehID = enforce_vehicle_format(uniqueVehicleIDs{vIdx});
    vehicleMetrics.(vehID) = struct( ...
        'macAddress', generate_mac_address(vIdx), ...
        'ipAddress', generate_ip_address(vIdx, simulationParams.numVehicles), ...
        'totalTx', 0, ...
        'successfulTx', 0, ...
        'latencies', [], ...
        'per', [], ...
        'fer', [], ...
        'ber', [], ...
        'cbr', [], ...
        'snr', [], ...
        'ser', [], ...
        'throughput', [], ...
        'calc_latency', [], ...
        'dataRates', [], ...
        'pdr', [], ...
        'transmissionPower', phyParams.transmissionPower, ...
        'transmissionPowerBefore', phyParams.transmissionPower, ...
        'beaconRate', simulationParams.beaconRate, ...
        'MCS', phyParams.MCS, ...
        'neighbors_number', 0, ...
        'pingRate', simulationParams.beaconRate, ...
        'macTxSuccess', 0, ...
        'macRetries', 0, ...
        'macDrops', 0, ...
        'macDelay', [], ...
        'macLatency', [], ...
        'macThroughput', [], ...
        'safetyTxCount', 0, ...
        'safetySuccessTx', 0 ...
    );

    fprintf('[MAC/IP] %s → MAC: %s | IP: %s\n', ...
    vehID, ...
    vehicleMetrics.(vehID).macAddress, ...
    vehicleMetrics.(vehID).ipAddress);
end
disp('[INFO] Vehicle metrics initialized successfully.');

% Sanity Check: Preview one initialized vehicle's metrics
disp('Debugging uniqueVehicleIDs:');
disp(uniqueVehicleIDs);

disp('Debugging vehicleMetrics fields:');
disp(fieldnames(vehicleMetrics)); % Display all initialized fields

vehID = enforce_vehicle_format(uniqueVehicleIDs{1});
if isfield(vehicleMetrics, vehID)
    disp(vehicleMetrics.(vehID)); % Preview first vehicle's metrics
else
    error('Field "%s" does not exist in vehicleMetrics.', vehID);
end

if isfield(vehicleMetrics, uniqueVehicleIDs{1})
    disp(vehicleMetrics.(uniqueVehicleIDs{1}));
else
    error('Field "%s" does not exist in vehicleMetrics.', uniqueVehicleIDs{1});
end

disp('[DEBUG] Initialized vehicleMetrics fields:');
disp(fieldnames(vehicleMetrics));

% Initialize beaconLog to store communication logs
% Each row in beaconLog will contain:
% [sourceID, targetID, timestamp, latency, dataRate, PER, BER, CBR]
beaconLog = cell(0, 8); % Predefine the structure for better performance
disp('[DEBUG] Initialized beaconLog with predefined structure.');
disp(beaconLog); % Should display an empty cell array


% =========================================================================
% Simulation Execution with RL and PHY Layer Integration
% =========================================================================

disp('[INFO] Starting simulation with RL and PHY layer integration...');
safetyLog = cell(0,7); % [Time, ID, Throughput, Latency, CBR, SNR, PER]

% Load time points from the FCD data
timePoints = unique([fcdData.time]);

% Initialize RL buffer to collect all vehicle updates per timestamp
rlBuffer = struct(); 

for tIdx = 1:length(timePoints)
    currentTime = timePoints(tIdx);
    currentFrame = fcdData([fcdData.time] == currentTime);
    % [PHASE 1] PROCESS ALL COMMUNICATIONS FIRST --------------------------
    % Process all vehicles' communications using CURRENT parameters
    % Debugging Output for Simulation Time Alignment
    disp(['[DEBUG] Current Time: ', num2str(currentTime), ...
        ' | Vehicles in Frame: ', num2str(length(currentFrame))]);

    if isempty(currentFrame)
        warning('[WARNING] No vehicles found in current frame at time %.2f s.', currentTime);
        continue; % Skip processing if no vehicles
    end

    %% Initialize neighbors list for the current frame
    vehicleData = struct();
    for i = 1:length(currentFrame)
        if i > length(currentFrame)
            warning('[WARNING] Index %d exceeds array size for currentFrame.', i);
            continue;
        end

        % Create temporary structure for parallel processing
        vehicleData(i).vehID = enforce_vehicle_format(currentFrame(i).id);
        vehicleData(i).pos = [currentFrame(i).x, currentFrame(i).y];
        vehicleData(i).originalIdx = i;
    end

    % Process vehicles in parallel
    parfor i = 1:length(vehicleData)
        vehID = vehicleData(i).vehID;
        pos1 = vehicleData(i).pos;
        originalIdx = vehicleData(i).originalIdx;
        
        % Create local copy of metrics to avoid PARFOR issues
        vehicleMetricsLocal = struct();
        
        % Check if vehicle metrics exist
        if ~isfield(vehicleMetrics, vehID)
            warning('Field "%s" does not exist in vehicleMetrics. Skipping...', vehID);
            continue;
        end

        % Calculate communication range for the current vehicle
        P_t = 10^(vehicleMetrics.(vehID).transmissionPower / 10) / 1e3; % Transmission power in Watts
        G_t = phyParams.G_t; % Transmitter gain
        G_r = phyParams.G_r; % Receiver gain
        f = phyParams.frequency; % Frequency in Hz
        R_s = 10^(-90 / 10); % Receiver sensitivity in W (example: -90 dBm)
        alpha = 2.5; % Path loss exponent (example)
        commRange = calculate_comm_range(P_t, G_t, G_r, f, R_s, alpha);
        vehicleMetricsLocal.commRange = commRange;

        % Debugging: Output the calculated communication range
        fprintf('[DEBUG] Vehicle %s | Communication Range: %.2f meters\n', vehID, commRange);

        % Neighbor detection with debugging
        neighbors = []; % Reset neighbors list for the current vehicle
        for j = 1:length(currentFrame)
            if originalIdx == j, continue; end % Skip self-comparison
            
            % Sanitize and validate neighbor ID
            otherVehID = enforce_vehicle_format(currentFrame(j).id);
            pos2 = [currentFrame(j).x, currentFrame(j).y]; % Position of neighbor
            
            % Calculate distance
            distance = calculate_distance(pos1, pos2);
            
            % Debugging: Log distances and positions
            fprintf('[DEBUG] Distance between %s and %s: %.2f m, Communication Range: %.2f m\n', ...
                vehID, otherVehID, distance, commRange);
            
            % Add neighbors if within communication range
            if distance <= commRange
                % Retrieve transmission power of the neighbor
                if isfield(vehicleMetrics, otherVehID)
                    neighborTxPower = vehicleMetrics.(otherVehID).transmissionPower; % In dBm
                else
                    neighborTxPower = phyParams.transmissionPower; % Default transmission power
                end
                
                % Add neighbor to the list
                neighbors = [neighbors; struct( ...
                    'id', otherVehID, ...
                    'distance', distance, ...
                    'P_tx', neighborTxPower ...
                )];
                
                % Remove duplicate neighbors
                [~, uniqueIdx] = unique({neighbors.id}, 'stable');
                neighbors = neighbors(uniqueIdx);
            end
        end
        
        % Store neighbors in local metrics
        vehicleMetricsLocal.neighbors = neighbors;
        
        % Debugging: Log neighbors detected
        if isempty(neighbors)
            fprintf('[DEBUG] No neighbors found for vehicle %s at time %.2f.\n', vehID, currentTime);
        else
            fprintf('[DEBUG] Neighbors detected for vehicle %s:\n', vehID);
            for k = 1:numel(neighbors)
                fprintf('  Neighbor: %s | Distance: %.2f m\n', neighbors(k).id, neighbors(k).distance);
            end
        end

        % Store the local metrics back to the main array
        vehicleData(i).metrics = vehicleMetricsLocal;
    end

    % Merge parallel results back into main vehicleMetrics structure
    for i = 1:length(vehicleData)
        vehID = vehicleData(i).vehID;
        if isfield(vehicleData(i), 'metrics')
            % Store neighbors in vehicleMetrics
            if ~isfield(vehicleMetrics, vehID)
                vehicleMetrics.(vehID) = struct();
            end
            vehicleMetrics.(vehID).neighbors = vehicleData(i).metrics.neighbors;
            vehicleMetrics.(vehID).commRange = vehicleData(i).metrics.commRange;
        end
    end

    % Continue with the rest of the processing (non-parallel part)
    for i = 1:length(currentFrame)
        vehID = enforce_vehicle_format(currentFrame(i).id);
        
        % Perform PHY Layer Communication to Update CBR and SNR
        sourceID = vehID; % Assign the current vehicle as the source
        if ~isfield(vehicleMetrics, vehID) || isempty(vehicleMetrics.(vehID).neighbors)
            warning('[WARNING] No neighbors found for vehicle %s at time %.2f.', vehID, currentTime);
            vehicleMetrics.(vehID).snr(end + 1) = NaN;
            continue; % skip ALL transmission
        end

        try
            %% Loop for beacon rate
            for beaconIdx = 1:1
                for k = 1:numel(vehicleMetrics.(vehID).neighbors)
                    neighbors = vehicleMetrics.(vehID).neighbors;
                    if ~isfield(neighbors(k), 'id') || ~isfield(neighbors(k), 'distance') || ...
                            isempty(neighbors(k).id) || isempty(neighbors(k).distance)
                        warning('[WARNING] Invalid neighbor structure for vehicle %s.', vehID);
                        continue;
                    end


                    % === Doppler Shift Initialization ===
                    if ~isfield(neighbors, 'dopplerShift')
                        warning('[INFO] Doppler shift not defined in neighbors. Defaulting to 0.');
                        for i = 1:length(neighbors)
                            neighbors(i).dopplerShift = 0;
                        end
                    end
                    
                    % === Signed Doppler Shift Calculation ===
                    for k = 1:length(neighbors)
                        targetVeh = neighbors(k).id;
                    
                        thisVehData = currentFrame(strcmp({currentFrame.id}, vehID));
                        targetVehData = currentFrame(strcmp({currentFrame.id}, targetVeh));
                    
                        if isempty(thisVehData) || isempty(targetVehData)
                            warning('[DOPPLER] Could not find vehicle data for %s or %s at t=%.2f', vehID, targetVeh, currentTime);
                            continue;
                        end
                    
                        % Velocity vectors
                        v1 = thisVehData.speed * [cosd(thisVehData.angle), sind(thisVehData.angle)];
                        v2 = targetVehData.speed * [cosd(targetVehData.angle), sind(targetVehData.angle)];
                        relativeVelocity = v1 - v2;
                    
                        % Relative position vector
                        pos1 = [thisVehData.x, thisVehData.y];
                        pos2 = [targetVehData.x, targetVehData.y];
                        relPos = pos1 - pos2;
                    
                        if norm(relPos) == 0
                            angleFactor = 1; % Assume max effect
                        else
                            relDir = relPos / norm(relPos);
                            angleFactor = dot(relativeVelocity, relDir) / norm(relativeVelocity + eps);
                        end
                    
                        % Doppler calculation
                        c = 3e8;
                        fc = phyParams.frequency;
                        relativeSpeed = norm(relativeVelocity);
                        signedDoppler = (relativeSpeed / c) * fc * angleFactor;
                    
                        neighbors(k).dopplerShift = signedDoppler;
                    end



                    % Dynamically generate random bits based on phyParams.payloadLength
                    payloadLengthBytes = phyParams.payloadLength; % Ensure payloadLength is in bytes
                    bitLength = payloadLengthBytes * 8; % Convert bytes to bits
                    customBits = randi([0 1], 1, bitLength); % Generate random bits (1 row, bitLength columns)


                   %% --- MAC performance logic --- 
                    maxRetries = macParams.retryLimit;
                    retryCount = 0;
                    macSuccess = false;
                    macDelay = 0;
                    totalDelay = 0;
                    macThru = 0;
                    attemptCount = 0; % Track total attempts (1 initial + retries)
                    
                    % Initialize all required MAC metrics with proper scaling factors
                    requiredMacFields = {'macThroughput', 'macLatency', 'macDelay', 'macRetries', ...
                                        'macSuccess', 'macDrops', 'macTotalAttempts', ...
                                        'macLatencyByTime', 'macThroughputByTime', ...
                                        'macRetriesByTime', 'macDropsByTime', 'macSuccessByTime'};
                    
                    % Initialize metrics with proper types
                    for f = 1:numel(requiredMacFields)
                        if ~isfield(vehicleMetrics.(vehID), requiredMacFields{f})
                            if contains(requiredMacFields{f}, 'ByTime')
                                vehicleMetrics.(vehID).(requiredMacFields{f}) = containers.Map('KeyType', 'double', 'ValueType', 'any');
                            elseif contains(requiredMacFields{f}, {'macRetries', 'macSuccess', 'macDrops', 'macTotalAttempts'})
                                vehicleMetrics.(vehID).(requiredMacFields{f}) = 0; % Initialize counters to 0
                            else
                                vehicleMetrics.(vehID).(requiredMacFields{f}) = []; % Initialize arrays as empty
                            end
                        end
                    end
                    
                    % Initialize timestamp containers if they don't exist
                    if ~isKey(vehicleMetrics.(vehID).macLatencyByTime, currentTime)
                        vehicleMetrics.(vehID).macLatencyByTime(currentTime) = [];
                    end
                    if ~isKey(vehicleMetrics.(vehID).macThroughputByTime, currentTime)
                        vehicleMetrics.(vehID).macThroughputByTime(currentTime) = [];
                    end
                    if ~isKey(vehicleMetrics.(vehID).macRetriesByTime, currentTime)
                        vehicleMetrics.(vehID).macRetriesByTime(currentTime) = 0;
                    end
                    if ~isKey(vehicleMetrics.(vehID).macDropsByTime, currentTime)
                        vehicleMetrics.(vehID).macDropsByTime(currentTime) = 0;
                    end
                    if ~isKey(vehicleMetrics.(vehID).macSuccessByTime, currentTime)
                        vehicleMetrics.(vehID).macSuccessByTime(currentTime) = 0;
                    end
                    
                    % Calculate dynamic collision probability based on neighbors and CBR
                    neighborCount = numel(vehicleMetrics.(vehID).neighbors);
                    currentCBR = mean(vehicleMetrics.(vehID).cbr, 'omitnan');
                    if isnan(currentCBR), currentCBR = 0; end
                    
                    % Base collision probability (5%) + 1% per neighbor + 20% of CBR
                    collision_probability = min(0.5, 0.05 + 0.01*neighborCount + 0.2*currentCBR);
                    vehicleMetrics.(vehID).macTotalAttempts = vehicleMetrics.(vehID).macTotalAttempts + 1;
                    
                    % MAC transmission attempt loop with realistic backoff
                    while attemptCount <= maxRetries
                        attemptCount = attemptCount + 1;
                        
                        % Perform PHY communication and get detailed metrics
                        phyOutput = perform_phy_comm(vehID, sourceID, '', neighbors, vehicleMetrics, phyParams, simulationParams, cbrParams, customBits);
                        
                        % Calculate effective PER including both PHY errors and collisions
                        phyPER = min(max(phyOutput.errorMetrics.PER, 0), 1); % Raw PHY PER (clamped 0-1)
                        effective_PER = min(1, phyPER + collision_probability - (phyPER * collision_probability));
                        
                        % Calculate total frame transmission time (PHY + MAC overhead)
                        payloadBits = phyParams.payloadLength * 8;
                        mac_header_bits = macParams.macHeaderBytes * 8;
                        total_bits = payloadBits + mac_header_bits;
                        
                        % Get PHY time from actual PHY parameters
                        symbol_time = 4e-6; % OFDM symbol duration
                        bits_per_symbol = phyParams.c_d; % From PHY output
                        total_symbols = ceil(total_bits / bits_per_symbol);
                        phy_time = total_symbols * symbol_time;
                        
                        % Calculate total delay for this attempt
                        attemptDelay = macParams.difs + macDelay + phy_time;
                        if macParams.ackEnabled
                            attemptDelay = attemptDelay + macParams.sifs + macParams.ackTimeout;
                        end
                        
                        % Check transmission success - modified to be more deterministic
                        successThreshold = rand(); % Single random number for the entire attempt
                        if successThreshold > effective_PER % Success
                            macSuccess = true;
                            totalDelay = attemptDelay; % Total successful transmission time
                            
                            % Calculate goodput (effective throughput)
                            successfulBits = payloadBits * (1 - effective_PER);
                            macThru = successfulBits / totalDelay; % Raw throughput in bps
                            
                            % Apply efficiency factor based on congestion
                            efficiency_factor = 0.7 * (1 - (0.015 * neighborCount));
                            macThru = macThru * efficiency_factor;
                            
                            % Store metrics
                            vehicleMetrics.(vehID).macThroughput(end + 1) = macThru / 1e6; % Mbps
                            vehicleMetrics.(vehID).macLatency(end + 1) = totalDelay;
                            
                            % Update time-specific metrics
                            currentLatencies = vehicleMetrics.(vehID).macLatencyByTime(currentTime);
                            vehicleMetrics.(vehID).macLatencyByTime(currentTime) = [currentLatencies, totalDelay];
                            
                            currentThroughputs = vehicleMetrics.(vehID).macThroughputByTime(currentTime);
                            vehicleMetrics.(vehID).macThroughputByTime(currentTime) = [currentThroughputs, macThru / 1e6];
                            
                            % Update success counters
                            vehicleMetrics.(vehID).macSuccess = vehicleMetrics.(vehID).macSuccess + 1;
                            vehicleMetrics.(vehID).macSuccessByTime(currentTime) = ...
                                vehicleMetrics.(vehID).macSuccessByTime(currentTime) + 1;
                            
                            % Record actual retries used (not counting the successful attempt)
                            vehicleMetrics.(vehID).macRetries = vehicleMetrics.(vehID).macRetries + retryCount;
                            vehicleMetrics.(vehID).macRetriesByTime(currentTime) = ...
                                vehicleMetrics.(vehID).macRetriesByTime(currentTime) + retryCount;
                            
                            break; % Exit retry loop on success
                        else % Failure - prepare for retry
                            retryCount = retryCount + 1;
                            
                            % Exponential backoff with 802.11p rules
                            CW = min(macParams.cwMin * (2^min(retryCount, 6)), macParams.cwMax);
                            backoffSlots = randi([0, CW]);
                            macDelay = macDelay + backoffSlots * macParams.slotTime;
                            
                            % Increase collision probability after each retry (but cap at 70%)
                            collision_probability = min(0.7, collision_probability + 0.1);
                            
                            % Record delay for this failed attempt (with penalty)
                            vehicleMetrics.(vehID).macDelay(end + 1) = attemptDelay * 1.3; % 30% penalty for retries
                        end
                    end
                    
                    % Handle final failure case (max retries exceeded)
                    if ~macSuccess
                        vehicleMetrics.(vehID).macDrops = vehicleMetrics.(vehID).macDrops + 1;
                        vehicleMetrics.(vehID).macDropsByTime(currentTime) = ...
                            vehicleMetrics.(vehID).macDropsByTime(currentTime) + 1;
                        
                        % Count all attempts as retries (including the initial attempt)
                        vehicleMetrics.(vehID).macRetries = vehicleMetrics.(vehID).macRetries + retryCount;
                        vehicleMetrics.(vehID).macRetriesByTime(currentTime) = ...
                            vehicleMetrics.(vehID).macRetriesByTime(currentTime) + retryCount;
                    end
                    
                    % Update composite metrics with PHY+MAC combined effects
                    vehicleMetrics.(vehID).latencies = [vehicleMetrics.(vehID).latencies, totalDelay];
                    vehicleMetrics.(vehID).per = [vehicleMetrics.(vehID).per, effective_PER];
                    vehicleMetrics.(vehID).ser = [vehicleMetrics.(vehID).ser, phyOutput.errorMetrics.SER];
                    vehicleMetrics.(vehID).ber = [vehicleMetrics.(vehID).ber, phyOutput.errorMetrics.BER];
                    vehicleMetrics.(vehID).dataRates = [vehicleMetrics.(vehID).dataRates, macThru];
                    vehicleMetrics.(vehID).snr = [vehicleMetrics.(vehID).snr, phyOutput.effectiveSnr];
                    vehicleMetrics.(vehID).cbr = [vehicleMetrics.(vehID).cbr, phyOutput.totalCBR];
                    
                    fprintf('Vehicle %s | MAC Throughput: %.2f Mbps | MAC Latency: %.4f ms | Effective PER: %.2f | Attempts: %d (Retries: %d)\n', ...
                            vehID, macThru/1e6, totalDelay*1000, effective_PER, attemptCount, retryCount);
                    
                    % =============================================================
                    % Safety Message Transmission (Critical Beacon)
                    % =============================================================
                    if rand() <= simulationParams.safetyMsgProbability
                        try
                            % Backup original parameters
                            originalTxPower = vehicleMetrics.(vehID).transmissionPower;
                            originalMCS = vehicleMetrics.(vehID).MCS;
                            originalPayload = phyParams.payloadLength;
                            
                            % Apply safety message parameters
                            vehicleMetrics.(vehID).transmissionPower = originalTxPower + simulationParams.safetyTxPowerIncrease;
                            vehicleMetrics.(vehID).MCS = simulationParams.safetyMCS;
                            phyParams.payloadLength = simulationParams.safetyPayloadLength;
                            
                            % Generate safety message payload
                            safetyPayloadBits = randi([0 1], 1, simulationParams.safetyPayloadLength*8);
                            
                            % Get current neighbors from vehicleMetrics
                            currentNeighbors = vehicleMetrics.(vehID).neighbors;
                            
                            if ~isempty(currentNeighbors)
                                % Transmit safety message to all neighbors
                                phyOutput = perform_phy_comm(vehID, vehID, '', currentNeighbors,...
                                                           vehicleMetrics, phyParams, simulationParams,...
                                                           cbrParams, safetyPayloadBits);
                                
                                % Log safety message metrics
                                vehicleMetrics.(vehID).safetyTxCount = vehicleMetrics.(vehID).safetyTxCount + 1;
                                if isfield(phyOutput, 'success') && phyOutput.success
                                    vehicleMetrics.(vehID).successfulTx = vehicleMetrics.(vehID).successfulTx + 1;
                                end
                                
                                % Update safety metrics
                                safetyLog(end+1,:) = {currentTime, vehID, phyOutput.dataRate,...
                                                    phyOutput.latency, phyOutput.totalCBR,...
                                                    phyOutput.effectiveSnr, phyOutput.errorMetrics.PER};
                            else
                                warning('[SAFETY] No neighbors for vehicle %s at t=%.2fs', vehID, currentTime);
                            end
                            
                        catch ME
                            warning('[SAFETY ERROR] Vehicle %s: %s', vehID, ME.message);
                        end
                        
                        % Restore original parameters
                        vehicleMetrics.(vehID).transmissionPower = originalTxPower;
                        vehicleMetrics.(vehID).MCS = originalMCS;
                        phyParams.payloadLength = originalPayload;
                    end


                    % Log successful transmissions
                    vehicleMetrics.(vehID).totalTx = vehicleMetrics.(vehID).totalTx + 1;
                    if isfield(phyOutput, 'success') && phyOutput.success
                        vehicleMetrics.(vehID).successfulTx = vehicleMetrics.(vehID).successfulTx + 1;
                    end
                end
            end
        catch ME
            warning('[ERROR] PHY communication failed for vehicle %s: %s', vehID, ME.message);
            continue; % Skip further processing for this vehicle
        end
    end
    
   % [PHASE 2] RL OPTIMIZATION - BATCH COMMUNICATION ----------------------
    if simulationParams.enableRL
        % Initialize RL data collection structure
        rlPayload = struct();
        vehicleList = {};
        
        % 1. COLLECT CURRENT STATE FROM ALL VEHICLES
        for i = 1:length(currentFrame)
            vehID = enforce_vehicle_format(currentFrame(i).id);
            vehicleList{end+1} = vehID;
            
            % Initialize vehicle metrics if not present
            if ~isfield(vehicleMetrics, vehID)
                vehicleMetrics.(vehID) = struct(...
                    'transmissionPower', phyParams.transmissionPower, ...
                    'MCS', 0, ...
                    'beaconRate', simulationParams.defaultBeaconRate, ...
                    'snr', NaN, ...
                    'cbr', NaN, ...
                    'neighbors', []);
            end
            
            % Ensure we have valid values (handle NaN/empty cases)
            vm = vehicleMetrics.(vehID);
            if isempty(vm.neighbors)
                neighborCount = 0;
            else
                neighborCount = numel(vm.neighbors);
            end
            
            % Prepare vehicle data for RL
            rlPayload.(vehID) = struct(...
                'CBR', mean(vm.cbr, 'omitnan'), ...
                'SNR', mean(vm.snr, 'omitnan'), ...
                'neighbors', neighborCount, ...
                'transmissionPower', vm.transmissionPower, ...
                'MCS', vm.MCS, ...
                'beaconRate', vm.beaconRate, ...
                'timestamp', currentTime ...
            );
        end
        
        % 2. SEND BATCH TO RL SERVER
        fprintf('[RL COMM] Sending batch for %d vehicles at t=%.2f\n', length(vehicleList), currentTime);
        disp(jsonencode(rlPayload));
        
        try
            % Send complete batch and get responses
            rlResponse = communicate_with_rl(rlClient, rlPayload);
           
            fprintf('[RL COMM] Received response for t=%.2f:\n', currentTime);
            disp(jsonencode(rlResponse));
           
            
            % 3. PROCESS RESPONSE AND PREPARE NEXT TIMESTAMP PARAMETERS
            updatedVehicles = fieldnames(rlResponse.vehicles);
            for i = 1:length(updatedVehicles)
                vehID = updatedVehicles{i};
                
                if ~isfield(vehicleMetrics, vehID)
                    warning('Received update for unknown vehicle: %s', vehID);
                    continue;
                end
                
                % Get the new parameters from RL response
                newParams = rlResponse.vehicles.(vehID);
                
                % Validate and apply constraints
                newPower = bound_value(newParams.transmissionPower, ...
                                      0, 30);
                
                newMCS = round(bound_value(newParams.MCS, 0, 10));
                
                newBeacon = bound_value(newParams.beaconRate, ...
                                       1, ...
                                       20);
                
                % Store for next timestamp (using next_ prefix)
                vehicleMetrics.(vehID).transmissionPower = newPower;
                vehicleMetrics.(vehID).MCS = newMCS;
                vehicleMetrics.(vehID).beaconRate = newBeacon;
                
                % Store training metrics if available
                if isfield(newParams, 'training')
                    vehicleMetrics.(vehID).lastTraining = newParams.training;
                end
                
                fprintf('[RL UPDATE] %s at t=%.2f:\n', vehID, currentTime);
                fprintf('  Power: %.2f -> %.2f dBm\n', ...
                    vehicleMetrics.(vehID).transmissionPower, newPower);
                fprintf('  MCS: %d -> %d\n', ...
                    vehicleMetrics.(vehID).MCS, newMCS);
                fprintf('  Beacon: %.2f -> %.2f Hz\n', ...
                    vehicleMetrics.(vehID).beaconRate, newBeacon);
                
            end
            
        catch ME
            fprintf('[RL ERROR] Communication failed at t=%.2f: %s\n', currentTime, ME.message);
            fprintf('Error details:\n');
            disp(ME);
        end
    end
    
    % [PHASE 3] APPLY UPDATES FOR NEXT TIMESTAMP ---------------------------
    % This runs at the end of each timestamp processing, before moving to next timestamp
    
    % Get all vehicle IDs that need updates
    vehiclesToUpdate = {};
    for i = 1:length(currentFrame)
        vehID = enforce_vehicle_format(currentFrame(i).id);
        if isfield(vehicleMetrics.(vehID), 'next_transmissionPower')
            vehiclesToUpdate{end+1} = vehID;
        end
    end
    
    % Apply the updates for next timestamp
    for i = 1:length(vehiclesToUpdate)
        vehID = vehiclesToUpdate{i};
        vm = vehicleMetrics.(vehID);
        
        % Apply the prepared updates
        vehicleMetrics.(vehID).transmissionPower = vm.next_transmissionPower;
        vehicleMetrics.(vehID).MCS = vm.next_MCS;
        vehicleMetrics.(vehID).beaconRate = vm.next_beaconRate;
        
        % Clean up the temporary fields
        vehicleMetrics.(vehID) = rmfield(vm, {'next_transmissionPower', 'next_MCS', 'next_beaconRate'});
        
        
        fprintf('[RL APPLIED] %s parameters updated for next timestamp\n', vehID);
    end
    




    % Visualization 1 Trace of beacons
    figure(1);
    scatter([currentFrame.x], [currentFrame.y], 'filled');
    hold on;
    for j = 1:length(currentFrame)
        text(currentFrame(j).x + 5, currentFrame(j).y, currentFrame(j).id, 'FontSize', 8);
        sanitizedID = enforce_vehicle_format(currentFrame(j).id);
        commRange = vehicleMetrics.(sanitizedID).commRange;
        rectangle('Position', [currentFrame(j).x - commRange, ...
            currentFrame(j).y - commRange, ...
            2 * commRange, 2 * commRange], ...
            'Curvature', [1, 1], 'EdgeColor', 'cyan', 'LineStyle', '--');
    end
    xlabel('X Position (m)');
    ylabel('Y Position (m)');
    title(sprintf('Vehicle Positions and Beacon Trace at Time %.2f', currentTime));
    grid on;
    pause(0.1);

    % Visualization
    figure(3);
    cla; % Clear the axes to remove previous plots
    scatter([currentFrame.x], [currentFrame.y], 'filled'); % Plot current positions
    for j = 1:length(currentFrame)
        % Add vehicle IDs as text labels
        text(currentFrame(j).x + 5, currentFrame(j).y, currentFrame(j).id, 'FontSize', 8);
        
        % Draw communication range circles
        sanitizedID = enforce_vehicle_format(currentFrame(j).id);
        commRange = vehicleMetrics.(sanitizedID).commRange;
        rectangle('Position', [currentFrame(j).x - commRange, ...
                              currentFrame(j).y - commRange, ...
                              2 * commRange, 2 * commRange], ...
                  'Curvature', [1, 1], 'EdgeColor', 'cyan', 'LineStyle', '--');
    end
    xlabel('X Position (m)');
    ylabel('Y Position (m)');
    title(sprintf('Vehicle Positions at Time %.2f', currentTime));
    grid on;
    pause(0.1); % Pause to visualize the current frame
    end

% Close RL Socket
if simulationParams.enableRL
    clear rlClient;
    disp('[INFO] RL socket closed.');
end

% Summarize and Visualize Results
outputFile = 'simulation_results.xlsx';
save_simulation_results(vehicleMetrics, timePoints, outputFile);

% Helper function for value bounding
function val = bound_value(val, minVal, maxVal)
    val = max(minVal, min(maxVal, val));
end

% =========================================================================
% Update Vehicle Metrics with Averaged Results
% =========================================================================

% Iterate through each unique vehicle ID to calculate and update metrics
for vIdx = 1:length(uniqueVehicleIDs)
    vehID = enforce_vehicle_format(uniqueVehicleIDs{vIdx});
    metrics = vehicleMetrics.(vehID);

    if metrics.totalTx > 0
        % Calculate averages for individual metrics
        metrics.avgMacThroughput = mean(metrics.macThroughput, 'omitnan');
        metrics.avgMacLatency = mean(metrics.macDelay, 'omitnan');        % Average Mac Latency
        metrics.avgLatency = mean(metrics.latencies, 'omitnan');          % Average PHY Latency
        metrics.avgPER = mean(metrics.per, 'omitnan');                    % Average Packet Error Rate (PER)
        metrics.avgSER = mean(metrics.ser, 'omitnan');                    % Average Symbol Error Rate (SER)
        metrics.avgBER = mean(metrics.ber, 'omitnan');                    % Average Bit Error Rate (BER)
        metrics.avgCBR = mean(metrics.cbr, 'omitnan');                    % Average Channel Busy Ratio (CBR)
        metrics.avgThroughput = mean(metrics.dataRates, 'omitnan');       % Average Throughput (Mbps)
        if isfield(metrics, 'avgPER') && ~isnan(metrics.avgPER)
            metrics.avgPDR = 1 - metrics.avgPER;
        elseif metrics.totalTx > 0
            metrics.avgPDR = metrics.successfulTx / metrics.totalTx;
        else
            metrics.avgPDR = 0;
        end
        metrics.avgSNR = mean(metrics.snr, 'omitnan');                    % Average SNR
    else
        % Set defaults for vehicles with no transmissions
        metrics.avgMacThroughput = NaN;
        metrics.avgLatency = NaN;
        metrics.avgPER = NaN;
        metrics.avgSER = NaN;
        metrics.avgBER = NaN;
        metrics.avgCBR = NaN;
        metrics.avgThroughput = NaN;
        metrics.avgPDR = 0; % PDR is 0 if no transmissions
        metrics.avgSNR = NaN;
    end

    % Store updated metrics back into the vehicleMetrics structure
    vehicleMetrics.(vehID) = metrics;
    % Debugging output after updating vehicleMetrics
    fprintf('[DEBUG] Vehicle %s | Latency: %.4f | PER: %.4f | SER: %.4f | BER: %.4f | Throughput: %.4f\n', ...
        vehID, phyOutput.latency, phyOutput.errorMetrics.PER, phyOutput.errorMetrics.SER, phyOutput.errorMetrics.BER, phyOutput.dataRate);
end

% =========================================================================
% Compute Overall Averages Across All Vehicles
% =========================================================================

% List of metric fields to average across all vehicles
metricFields = {'avgLatency', 'avgPER', 'avgSER', 'avgBER', ...
                'avgCBR', 'avgThroughput', 'avgPDR', ...
                 'avgSNR','avgMacLatency', 'avgMacThroughput'};

% Initialize overallMetrics structure
overallMetrics = struct();

% Calculate overall averages for each metric field
for fieldIdx = 1:length(metricFields)
    field = metricFields{fieldIdx};
    fieldValues = []; % Initialize a container for field values
    for vIdx = 1:length(uniqueVehicleIDs)
        vehID = enforce_vehicle_format(uniqueVehicleIDs{vIdx}); % Ensure proper formatting
        if isfield(vehicleMetrics.(vehID), field) && ~isempty(vehicleMetrics.(vehID).(field))
            fieldValues = [fieldValues, vehicleMetrics.(vehID).(field)]; %#ok<AGROW>
        end
    end
    overallMetrics.(field) = mean(fieldValues, 'omitnan'); % Calculate mean
end

% Display overall metrics in the console
disp('[INFO] Overall Simulation Averages:');
disp(overallMetrics);

% =========================================================================
% Visualize Performance Metrics by Timestamp
% =========================================================================

% Initialize arrays to store metrics over time
timestamps = unique([fcdData.time]);
numTimestamps = length(timestamps);

% Preallocate arrays for overall averages at each timestamp
overallMetricsByTime = struct('avgLatency', zeros(1, numTimestamps), ...
                              'avgPER', zeros(1, numTimestamps), ...
                              'avgSER', zeros(1, numTimestamps), ...
                              'avgBER', zeros(1, numTimestamps), ...
                              'avgCBR', zeros(1, numTimestamps), ...
                              'avgThroughput', zeros(1, numTimestamps), ...
                              'avgPDR', zeros(1, numTimestamps), ...
                              'avgSNR', zeros(1, numTimestamps)); 

% Loop through each timestamp and calculate overall metrics
for tIdx = 1:numTimestamps
    currentTime = timestamps(tIdx);
    currentFrame = fcdData([fcdData.time] == currentTime);

    % Extract metrics for the current frame
    totalVehicles = 0;
    totalMetrics = struct('latencies', [], 'per', [], 'ser', [], 'ber', [], ...
                      'cbr', [], 'dataRates', [], 'snr', [], ...
                      'successfulTx', 0, 'totalTx', 0);


    for vIdx = 1:length(currentFrame)
        vehID = enforce_vehicle_format(currentFrame(vIdx).id);
        if isfield(vehicleMetrics, vehID)
            metrics = vehicleMetrics.(vehID);
            if metrics.totalTx > 0
                totalVehicles = totalVehicles + 1;

                % Accumulate metrics
                totalMetrics.latencies = [totalMetrics.latencies, metrics.latencies]; %#ok<AGROW>
                totalMetrics.per = [totalMetrics.per, metrics.per]; %#ok<AGROW>
                totalMetrics.ser = [totalMetrics.ser, metrics.ser]; %#ok<AGROW>
                totalMetrics.ber = [totalMetrics.ber, metrics.ber]; %#ok<AGROW>
                totalMetrics.cbr = [totalMetrics.cbr, metrics.cbr]; %#ok<AGROW>
                totalMetrics.dataRates = [totalMetrics.dataRates, metrics.dataRates]; %#ok<AGROW>
                totalMetrics.snr = [totalMetrics.snr, metrics.snr]; %#ok<AGROW>
                totalMetrics.successfulTx = totalMetrics.successfulTx + metrics.successfulTx;
                totalMetrics.totalTx = totalMetrics.totalTx + metrics.totalTx;
            end
        end
    end

    % Compute overall averages for the current timestamp
    overallMetricsByTime.avgLatency(tIdx) = mean(totalMetrics.latencies, 'omitnan');
    overallMetricsByTime.avgPER(tIdx) = mean(totalMetrics.per, 'omitnan');
    overallMetricsByTime.avgSER(tIdx) = mean(totalMetrics.ser, 'omitnan');
    overallMetricsByTime.avgBER(tIdx) = mean(totalMetrics.ber, 'omitnan');
    overallMetricsByTime.avgCBR(tIdx) = mean(totalMetrics.cbr, 'omitnan');
    overallMetricsByTime.avgThroughput(tIdx) = mean(totalMetrics.dataRates, 'omitnan');
    % overallMetricsByTime.avgPDR(tIdx) = totalMetrics.successfulTx / max(1, totalMetrics.totalTx); % Avoid division by zero
    overallMetricsByTime.avgPDR(tIdx) = 1-overallMetricsByTime.avgPER(tIdx);
    overallMetricsByTime.avgSNR(tIdx) = mean(totalMetrics.snr, 'omitnan');
    writetable(cell2table(safetyLog, 'VariableNames',...
    {'Timestamp','VehicleID','Throughput','Latency','CBR','SNR','PER'}),...
    'safety_log.csv');
end

% =========================================================================
% Plot Performance Metrics by Timestamp
% =========================================================================

figure('Name', 'Overall Simulation Metrics by Timestamp', 'NumberTitle', 'off');

% Plot average latency
subplot(3, 3, 1);
plot(timestamps, overallMetricsByTime.avgLatency, '-o');
title('Average Latency Over Time');
xlabel('Timestamp (s)');
ylabel('Latency (s)');
grid on;

% Plot average PER
subplot(3, 3, 2);
plot(timestamps, overallMetricsByTime.avgPER, '-o');
title('Average PER Over Time');
xlabel('Timestamp (s)');
ylabel('PER');
grid on;

% Plot average SER
subplot(3, 3, 3);
plot(timestamps, overallMetricsByTime.avgSER, '-o');
title('Average SER Over Time');
xlabel('Timestamp (s)');
ylabel('SER');
grid on;

% Plot average BER
subplot(3, 3, 4);
plot(timestamps, overallMetricsByTime.avgBER, '-o');
title('Average BER Over Time');
xlabel('Timestamp (s)');
ylabel('BER');
grid on;

% Plot average CBR
subplot(3, 3, 5);
plot(timestamps, overallMetricsByTime.avgCBR, '-o');
title('Average CBR Over Time');
xlabel('Timestamp (s)');
ylabel('CBR');
grid on;

% Plot average throughput
subplot(3, 3, 6);
plot(timestamps, overallMetricsByTime.avgThroughput, '-o');
title('Average Throughput Over Time');
xlabel('Timestamp (s)');
ylabel('Throughput (Mbps)');
grid on;

% Plot average PDR
subplot(3, 3, 7);
plot(timestamps, overallMetricsByTime.avgPDR, '-o');
title('Average PDR Over Time');
xlabel('Timestamp (ms)');
ylabel('PDR');
grid on;

% Plot average SNR
subplot(3, 3, 8);  % Gunakan slot kosong di grid 3x3
plot(timestamps, overallMetricsByTime.avgSNR, '-o');
title('Average SNR Over Time');
xlabel('Timestamp (s)');
ylabel('SNR (dB)');
grid on;


% Adjust layout for better visualization
sgtitle('Overall Simulation Metrics Over Time');


%% =========================================================================
% Supporting Functions
% =========================================================================


% =========================================================================
% CBR Calculation Functions
% =========================================================================
%function cbr = calculate_CBR_type1(r_cs, rho, b, C)
%    % CBR Type 1 Calculation
%    cbr = (2 * r_cs * rho * b) / C;
%    cbr = min(cbr, 1); % Cap at 1
%end

function cbr = calculate_CBR_type1(r_cs, numNeighbors, b, C, p, p_prime, beta, mcs, simParams)
% Calculate the Channel Busy Ratio (CBR) using the revised Type 1 formula.
%
% Inputs:
%   r_cs: Carrier sense range (meters)
%   numNeighbors: Number of neighbors within communication range
%   b: Beacon rate (beacons per second)
%   C: Channel capacity (symbols per second)
%   p: Current transmission power (Watts)
%   p_prime: New transmission power after adjustment (Watts)
%   beta: Path loss exponent
%   mcs: Modulation and Coding Scheme index (0–10)
%   simParams: struct or trafficProfile string for estimating bgCBR
%
% Output:
%   cbr: Channel Busy Ratio (dimensionless, capped at 1)

    % Step 1: Calculate vehicle density
    area = pi * r_cs^2;
    rho = numNeighbors / area;

    % Step 2: Adjusted neighbor count
    n_prime = rho * area * (p / p_prime)^(1 / beta);

    % Step 3: Adaptive Max Beacon Rate
    maxBeaconRate = calculate_max_beacon_rate(mcs, 350, simParams);
    %maxBeaconRate = 200;
    % Step 4: Estimate CBR
    cbr = ((n_prime + 1) * b) / maxBeaconRate;
    cbr = min(cbr, 1);
end



% CBR Type 2, is CBR Calculation based on MCS and Transmit Power Adaptation
function cbr = calculate_CBR_type2(r_cs, rho, b, C)
    % Calculate Channel Busy Ratio (CBR) using the revised Type 2 formula.
    %
    % Inputs:
    %   r_cs: Carrier sense range (meters)
    %   rho: Vehicle density (vehicles per square meter)
    %   b: Beacon rate (beacons per second)
    %   C: Channel capacity (symbols per second)
    %
    % Output:
    %   cbr: Channel Busy Ratio (dimensionless, capped at 1)

    % Calculate CBR
    cbr = (2 * r_cs * rho * b) / C;

    % Cap CBR at 1 (it cannot exceed 100%)
    cbr = min(cbr, 1);
end

function cbr = calculate_CBR_type3(phyParams, vehicleMetrics, cbrParams)
    % Calculate Channel Busy Ratio (CBR) using the enhanced Type 3 formula.
    %
    % Inputs:
    %   phyParams: Structure containing PHY layer parameters
    %   vehicleMetrics: Structure containing vehicle-specific metrics
    %   cbrParams: Structure containing additional CBR-related parameters
    %
    % Output:
    %   cbr: Channel Busy Ratio (dimensionless, capped at 1)

    % Extract necessary parameters
    bandwidth = phyParams.bandwidth; % Bandwidth in Hz
    noiseFigure = phyParams.noiseFigure; % Noise figure in dB
    P_tx = phyParams.transmissionPower; % Transmit power in dBm
    distance = 100; % Example distance in meters (adjust based on context)
    frequency = phyParams.frequency; % Frequency in Hz

    % Constants
    k = 1.38e-23; % Boltzmann constant (J/K)
    T = 290; % Room temperature (K)
    c = 3e8; % Speed of light (m/s)
    lambda = c / frequency; % Wavelength in meters

    % Noise Power Calculation (in dBm)
    P_noise_W = k * T * bandwidth; % Noise power in Watts
    P_noise_dBm = 10 * log10(P_noise_W * 1e3) + noiseFigure; % Convert to dBm, include noise figure

    % Path Loss Calculation (Free-space model)
    pathLoss_dB = 20 * log10(4 * pi * distance / lambda); % Free-space path loss in dB

    % Received Power Calculation (in dBm)
    P_rx_dBm = P_tx - pathLoss_dB; % Received power at the receiver (dBm)

    % Effective SNR Calculation (in dB)
    effectiveSnr = P_rx_dBm - P_noise_dBm; % SNR (dB) accounting for path loss and noise

    % Convert SNR from dB to linear scale
    SNR_linear = 10^(effectiveSnr / 10); % Linear scale SNR for further calculations

    % Calculate Channel Capacity (bps)
    ChannelCapacity = bandwidth * log2(1 + SNR_linear); % Shannon-Hartley formula

    % Retrieve MCS level and calculate spectral efficiency
    MCS = vehicleMetrics.(vehID).MCS; % Modulation and Coding Scheme level
    MCS_factor = get_MCS_factor(MCS); % Spectral efficiency (bits/symbol)

    % Calculate CBR
    cbr = (vehicleMetrics.transmissionPower * MCS_factor * vehicleMetrics.beaconRate) ...
        / (ChannelCapacity * pathLoss_dB);

    % Cap CBR at 1 (it cannot exceed 100%)
    cbr = min(cbr, 1);
end

function r_cs = calculate_r_cs(S, p, beta, m, wavelength)
    % Calculate the carrier sense range (r_CS) using the given formula.
    %
    % Inputs:
    %   S: Receiver sensitivity (Watts)
    %   p: Transmission power (Watts)
    %   beta: Path loss exponent
    %   m: Shape parameter indicating the severity of fading
    %   wavelength: Wavelength of the carrier signal (meters)
    %
    % Output:
    %   r_cs: Carrier sense range (meters)

    % Constants
    A = (4 * pi / wavelength)^2; % Factor dependent on wavelength

    % Calculate r_CS
    numerator = gamma(m + 1 / beta); % Gamma function term
    denominator = gamma(m) * (S * A * p); % Denominator terms
    r_cs = (numerator / denominator)^(1 / beta);

    % Ensure r_cs is non-negative
    r_cs = max(r_cs, 0);
end


function C = calculate_C(c_d, b_st, M, t_ps)
    % Calculate channel capacity (C) using MCS-dependent parameters.
    %
    % Inputs:
    %   c_d: Number of coded bits per OFDM symbol (dependent on MCS)
    %   b_st: Service and tail bits (e.g., 22 bits)
    %   M: Packet length in bits (e.g., 536 bytes converted to bits)
    %   t_ps: Preamble/signal field duration (seconds, e.g., 40 µs)
    %
    % Output:
    %   C: Channel capacity (symbols per second)

    % Calculate the denominator
    denominator = ceil((b_st + M) / c_d) + t_ps;

    % Calculate channel capacity
    C = (c_d / denominator)^(-1); % Inverse to get symbols per second
end


function MCS_factor = get_MCS_factor(MCS)
    % Define spectral efficiency for MCS levels
    %
    % Input:
    %   MCS: Modulation and Coding Scheme level (integer between 0 and 10)
    %
    % Output:
    %   MCS_factor: Spectral efficiency (bits/symbol)

    % Define spectral efficiency for MCS levels
    spectralEfficiency = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 4.5, 5.0, 6.0, 6.5, 3.0]; % 0-10

    if MCS >= 0 && MCS < length(spectralEfficiency)
        MCS_factor = spectralEfficiency(MCS + 1); % MCS starts from 0
    else
        error('Invalid MCS level: %d. MCS must be between 0 and %d.', MCS, length(spectralEfficiency) - 1);
    end
end

% Define the mapping between MCS and c_d
function c_d = mcsToCdMapping(mcsValue)
    % Define spectral efficiency for MCS levels
    mapping = [12, 18, 24, 36, 48, 54, 64, 72, 96, 108, 128]; % Example values for c_d
    if mcsValue >= 0 && mcsValue <= length(mapping) - 1
        c_d = mapping(mcsValue + 1); % MCS starts from 0
    else
        error('Invalid MCS level: %d. MCS must be between 0 and %d.', mcsValue, length(mapping) - 1);
    end
end

function formattedID = enforce_vehicle_format(rawID)
    % Fungsi untuk memastikan ID kendaraan memiliki format yang benar
    if isnumeric(rawID)
        formattedID = sprintf('veh%d', rawID);
    elseif ischar(rawID) || isstring(rawID)
        rawID = regexprep(char(rawID), '\.\d+$', ''); % Hilangkan angka desimal
        if startsWith(rawID, 'veh')
            formattedID = rawID; % Jika sudah diawali 'veh', biarkan
        else
            formattedID = sprintf('veh%s', rawID); % Tambahkan 'veh' jika belum ada
        end
    else
        warning('[WARNING] Unrecognized vehicle ID format.');
        formattedID = '';
    end
end

%{
function maxBeaconRate = calculate_max_beacon_rate(mcsValue, payloadSizeBytes)
    % calculate_max_beacon_rate - Calculates the maximum beacon rate per second.
    %
    % Inputs:
    %   mcsValue - Modulation and Coding Scheme (MCS) index (integer, typically 0-10).
    %   payloadSizeBytes - Payload size in bytes (integer).
    %
    % Outputs:
    %   maxBeaconRate - Maximum beacon rate in Hz (integer).

    % Define constants for IEEE 802.11bd with 20 MHz bandwidth
    BANDWIDTH = 20e6; % Bandwidth in Hz
    SYMBOL_DURATION = 4e-6; % OFDM symbol duration in seconds (20 MHz bandwidth)
    GUARD_INTERVAL = 0.8e-6; % Guard interval in seconds
    FFT_SIZE = 64; % Number of subcarriers in FFT
    DATA_SUBCARRIERS = 52; % Number of data subcarriers

    % Mapping of MCS to modulation order (M) and coding rate
    mcsTable = [
        0,  2,  1/2;   % MCS 0  -> BPSK (M = 2), Code Rate = 1/2
        1,  4,  1/2;   % MCS 1  -> QPSK (M = 4), Code Rate = 1/2
        2,  4,  3/4;   % MCS 2  -> QPSK (M = 4), Code Rate = 3/4
        3, 16,  1/2;   % MCS 3  -> 16-QAM (M = 16), Code Rate = 1/2
        4, 16,  3/4;   % MCS 4  -> 16-QAM (M = 16), Code Rate = 3/4
        5, 64,  2/3;   % MCS 5  -> 64-QAM (M = 64), Code Rate = 2/3
        6, 64,  3/4;   % MCS 6  -> 64-QAM (M = 64), Code Rate = 3/4
        7, 64,  5/6;   % MCS 7  -> 64-QAM (M = 64), Code Rate = 5/6
        8, 256, 3/4;   % MCS 8  -> 256-QAM (M = 256), Code Rate = 3/4
        9, 256, 5/6;   % MCS 9  -> 256-QAM (M = 256), Code Rate = 5/6
       10, 1024, 3/4;  % MCS 10 -> 1024-QAM (M = 1024), Code Rate = 3/4
    ];

    % Validate MCS value
    if mcsValue < 0 || mcsValue > size(mcsTable, 1) - 1
        error('Invalid MCS value. Must be between 0 and %d.', size(mcsTable, 1) - 1);
    end

    % Extract modulation order (M) and coding rate for the given MCS
    modulationOrder = mcsTable(mcsValue + 1, 2); % Modulation order (M)
    codingRate = mcsTable(mcsValue + 1, 3);      % Coding rate (R)

    % Calculate bits per symbol
    bitsPerSymbol = log2(modulationOrder); % Bits per symbol

    % Calculate data rate (in bps)
    dataRateBps = BANDWIDTH * DATA_SUBCARRIERS * bitsPerSymbol * codingRate / FFT_SIZE;

    % Convert payload size to bits
    payloadSizeBits = payloadSizeBytes * 8;

    % Calculate maximum beacon rate (in Hz)
    maxBeaconRate = floor(dataRateBps / payloadSizeBits);

    % Display the result for debugging purposes
    fprintf('\n'); 
    fprintf('MCS Value: %d -> Data Rate: %.2f Mbps -> Max Beacon Rate: %d Hz\n', ...
        mcsValue, dataRateBps / 1e6, maxBeaconRate);
end
%}

function maxBeaconRate = calculate_max_beacon_rate(mcsValue, payloadSizeBytes, simulationParams)
% calculate_max_beacon_rate - Estimates practical max beacon rate (Hz)
% considering PHY rate, MAC overhead, and background CBR load.
%
% Inputs:
%   mcsValue           - MCS index (0–10)
%   payloadSizeBytes   - Beacon payload size in bytes
%   simulationParams   - Structure with fields: bgCBRFluctuation or trafficProfile
%
% Output:
%   maxBeaconRate      - Estimated max beacon rate per node (Hz)

    % === PHY Parameters (IEEE 802.11bd, 20 MHz)
    BANDWIDTH = 20e6;
    FFT_SIZE = 64;
    DATA_SUBCARRIERS = 52;
    SYMBOL_DURATION = 4e-6;
    GUARD_INTERVAL = 0.8e-6;

    % === MAC Timing Parameters
    SLOT_TIME = 13e-6;
    DIFS = 32e-6;
    CW_MIN = 15;
    AVG_BACKOFF = (CW_MIN / 2) * SLOT_TIME;

    % === MCS Table [MCS, ModOrder, CodeRate]
    mcsTable = [
        0,  2,  1/2;
        1,  4,  1/2;
        2,  4,  3/4;
        3, 16,  1/2;
        4, 16,  3/4;
        5, 64,  2/3;
        6, 64,  3/4;
        7, 64,  5/6;
        8, 256, 3/4;
        9, 256, 5/6;
       10,1024, 3/4
    ];

    if mcsValue < 0 || mcsValue > 10
        error('Invalid MCS value. Must be in range 0–10.');
    end
    modOrder = mcsTable(mcsValue + 1, 2);
    codeRate = mcsTable(mcsValue + 1, 3);
    bitsPerSymbol = log2(modOrder);

    % === PHY bitrate (bps)
    phyRate = BANDWIDTH * (DATA_SUBCARRIERS / FFT_SIZE) * bitsPerSymbol * codeRate;

    % === Airtime per beacon
    payloadBits = payloadSizeBytes * 8;
    txTimePHY = payloadBits / phyRate;
    txTimeMAC = DIFS + AVG_BACKOFF;
    packetAirTime = txTimePHY + txTimeMAC;

    % === Background CBR estimation ===
    if isfield(simulationParams, 'trafficProfile')
        switch simulationParams.trafficProfile
            case 'awgn_sparse'
                mu = 0.15; sigma = 0.02;
            case 'rural_los'
                mu = 0.30; sigma = 0.03;
            case 'highway_los'
                mu = 0.45; sigma = 0.04;
            case 'highway_nlos'
                mu = 0.60; sigma = 0.05;
            case 'urban_los'
                mu = 0.70; sigma = 0.06;
            case 'dense_urban'
                mu = 0.80; sigma = 0.05;
            case 'max_density'
                mu = 0.90; sigma = 0.03;
            otherwise
                error('Invalid trafficProfile.');
        end
        bgCBR_sampled = normrnd(mu, sigma);
        bgCBR_mean = min(max(bgCBR_sampled, 0), 1);
    elseif isfield(simulationParams, 'bgCBRFluctuation')
        bgCBR_mean = mean(simulationParams.bgCBRFluctuation);
    else
        error('simulationParams must include either trafficProfile or bgCBRFluctuation');
    end

    % === Adaptive scaling factor based on background CBR ===
    alpha = 1.0;
    beta = 2.0;
    scalingFactor = alpha / (1 + beta * bgCBR_mean);

    usableBeaconCBR = max(0, 1 - bgCBR_mean);
    maxBeaconRate = floor((usableBeaconCBR / packetAirTime) * scalingFactor /2.5);

    % === Debug Info
    fprintf('[INFO] MCS %d | PHY Rate: %.2f Mbps | Packet Airtime: %.2f µs\n', ...
        mcsValue, phyRate / 1e6, packetAirTime * 1e6);
    fprintf('[INFO] Background CBR ~ %.2f | Scaling Factor = %.2f → Max Beacon Rate: %d Hz\n', ...
        bgCBR_mean, scalingFactor, maxBeaconRate);
end


% =========================================================================
% Pathloss, Distance, Vehicle IPs  Function
% =========================================================================


function pathLoss = path_loss_model(distance)
    % Log-distance path loss model
    pathLossExponent = 2.5; % Adjust based on environment
    referenceLoss = 30;     % dB at 1 meter
    distance = max(distance, 1); % Avoid log(0)
    pathLoss = referenceLoss + 10 * pathLossExponent * log10(distance);
end

function distance = calculate_distance(pos1, pos2)
    % Calculate Euclidean distance
    distance = sqrt((pos1(1) - pos2(1))^2 + (pos1(2) - pos2(2))^2);
end

function mac = generate_mac_address(index)
    % Generate a MAC address using the vehicle index
    mac = sprintf('00:16:3E:%02X:%02X:%02X', ...
        bitshift(index, -8), bitand(index, 255), randi([0, 255]));
end

function ip = generate_ip_address(index, totalVehicles)
    % Generate IP address per vehicle based on subnet layout
    % Format: 192.168.SUBNET.HOST where subnet changes every 255
    subnet = floor((index - 1) / 255);
    host = mod((index - 1), 255) + 1;
    ip = sprintf('192.168.%d.%d', subnet, host);
end


% =========================================================================
% Summarize Communication Result
% =========================================================================


function summarize_communication(beaconLog, simulationDuration, vehicleMetrics)
    if isempty(beaconLog)
        disp('[INFO] No communication data logged.');
        return;
    end
    avgLatency = mean(cell2mat(beaconLog(:, 4)));
    avgDataRate = mean(cell2mat(beaconLog(:, 5)));

    disp(['[INFO] Simulation Results:']);
    disp([' - Avg Latency: ', num2str(avgLatency), ' s']);
    disp([' - Avg Data Rate: ', num2str(avgDataRate), ' Kb/s']);

    % Display per-vehicle metrics
    disp('[INFO] Per-Vehicle Metrics:');
    vehicleIDs = fieldnames(vehicleMetrics);
    for i = 1:numel(vehicleIDs)
        vehID = vehicleIDs{i};
        metrics = vehicleMetrics.(vehID);
        if metrics.totalTx > 0
            % pdr = metrics.successfulTx / metrics.totalTx;
            avgLat = mean(metrics.latencies);
            avgPER = mean(metrics.per);
            pdr = 1-avgPER;
            avgFER = mean(metrics.fer);
            avgBER = mean(metrics.ber);
            avgRate = mean(metrics.dataRates);
            avgCBR = mean(metrics.cbr);
            disp(['   Vehicle ', vehID, ...
                  ' | PDR: ', num2str(pdr * 100, '%.2f'), '%', ...
                  ' | Avg Latency: ', num2str(avgLat, '%.4f'), ' s', ...
                  ' | Avg PER: ', num2str(avgPER), ...
                  ' | Avg FER: ', num2str(avgFER), ...
                  ' | Avg BER: ', num2str(avgBER), ...
                  ' | Avg Data Rate: ', num2str(avgRate), ' Kb/s', ...
                  ' | Avg CBR: ', num2str(avgCBR)]);
        else
            disp(['   Vehicle ', vehID, ' | No transmissions attempted.']);
        end
    end
end

% =========================================================================
% FCD Data Load
% =========================================================================


%Load FCD Data Type 1
function fcdData = load_fcd_data(filePath)
    docNode = xmlread(filePath);
    timesteps = docNode.getElementsByTagName('timestep');
    fcdData = [];

    for tIdx = 0:timesteps.getLength-1
        timestep = timesteps.item(tIdx);
        time = str2double(timestep.getAttribute('time'));
        vehicles = timestep.getElementsByTagName('vehicle');

        for vIdx = 0:vehicles.getLength-1
            vehicle = vehicles.item(vIdx);
            vehID = char(vehicle.getAttribute('id'));
            x = str2double(vehicle.getAttribute('x'));
            y = str2double(vehicle.getAttribute('y'));
            fcdData = [fcdData; struct('id', vehID, 'time', time, 'x', x, 'y', y)];
        end
    end
end

% Load FCD Data Type 2
function fcdData = load_fcd_data_type2(filePath)
    % Load FCD data from XML file
    try
        docNode = xmlread(filePath);
        timesteps = docNode.getElementsByTagName('timestep');
        if timesteps.getLength == 0
            error('[ERROR] No <timestep> elements found in the XML file.');
        end
        
        fcdData = [];
        
        % Loop through each timestep
        for tIdx = 0:timesteps.getLength-1
            timestep = timesteps.item(tIdx);
            
            % Extract and validate the "time" attribute
            timeAttr = timestep.getAttribute('time');
            if isempty(timeAttr) || ~ischar(timeAttr)
                warning('[WARNING] Invalid or missing "time" attribute in timestep %d. Skipping.', tIdx);
                continue;
            end
            
            time = str2double(timeAttr);
            if isnan(time)
                warning('[WARNING] Invalid "time" value: "%s". Skipping timestep.', timeAttr);
                continue;
            end
            
            % Get all vehicles in the current timestep
            vehicles = timestep.getElementsByTagName('vehicle');
            if vehicles.getLength == 0
                warning('[WARNING] No <vehicle> elements found in timestep at time %.2f. Skipping.', time);
                continue;
            end
            
            % Process each vehicle
            for vIdx = 0:vehicles.getLength-1
                vehicle = vehicles.item(vIdx);
                
                % Extract and validate attributes
                vehID = char(vehicle.getAttribute('id'));
                xAttr = vehicle.getAttribute('x');
                yAttr = vehicle.getAttribute('y');
                
                if isempty(vehID) || ~ischar(vehID)
                    warning('[WARNING] Missing or invalid "id" attribute in vehicle at time %.2f. Skipping.', time);
                    continue;
                end
                
                x = str2double(xAttr);
                y = str2double(yAttr);
                
                if isnan(x) || isnan(y)
                    warning('[WARNING] Invalid "x" or "y" value for vehicle %s at time %.2f. Skipping.', vehID, time);
                    continue;
                end
                
                % Append valid vehicle data
                fcdData(end + 1) = struct('id', vehID, 'time', time, 'x', x, 'y', y);
            end
        end
        
        % Final check: Ensure fcdData is not empty
        if isempty(fcdData)
            error('[ERROR] No valid FCD data was loaded from the XML file.');
        end
        
        disp('[INFO] FCD data loaded successfully.');
    catch ME
        error('[ERROR] Failed to load FCD data: %s', ME.message);
    end
end

% Load FCD Data Type 3
function fcdData = load_fcd_data_type3(filePath, startTime, endTime)
    % Load FCD data from an XML file and filter by start and end time.
    %
    % Inputs:
    %   filePath - Path to the FCD XML file.
    %   startTime - Start time for filtering (in seconds).
    %   endTime - End time for filtering (in seconds).
    %
    % Outputs:
    %   fcdData - Filtered FCD data as an array of structs.

    % Read the XML file
    docNode = xmlread(filePath);
    timesteps = docNode.getElementsByTagName('timestep');
    
    % Initialize output structure
    fcdData = [];
    
    % Loop through all timesteps in the XML file
    for tIdx = 0:timesteps.getLength-1
        timestep = timesteps.item(tIdx);
        time = str2double(timestep.getAttribute('time')); % Extract time
        
        % Check if the current timestep is within the specified time range
        if time >= startTime && time <= endTime
            vehicles = timestep.getElementsByTagName('vehicle');
            
            % Loop through all vehicles in the current timestep
            for vIdx = 0:vehicles.getLength-1
                vehicle = vehicles.item(vIdx);
                vehID = char(vehicle.getAttribute('id')); % Vehicle ID
                x = str2double(vehicle.getAttribute('x')); % X-coordinate
                y = str2double(vehicle.getAttribute('y')); % Y-coordinate
                speed = str2double(vehicle.getAttribute('speed'));
                angle = str2double(vehicle.getAttribute('angle'));
                % Append vehicle data to the output structure
                fcdData = [fcdData; struct('id', vehID, 'time', time, 'x', x, 'y', y, ...
                           'speed', speed, 'angle', angle)];
            end
        end
    end
    
    % Display summary of loaded data
    disp(['[INFO] Loaded FCD data with ', num2str(length(fcdData)), ' entries between time ', ...
          num2str(startTime), ' and ', num2str(endTime), ' seconds.']);
end

function plot_statistics(vehicleMetrics, simulationParams, avgMCS)
    % Extract data from metrics
    vehicleIDs = fieldnames(vehicleMetrics);
    numVehicles = numel(vehicleIDs);
    
    % Preallocate arrays for overall statistics
    avgLatencies = zeros(1, numVehicles);
    avgPERs = zeros(1, numVehicles);
    avgFERs = zeros(1, numVehicles);
    avgBERs = zeros(1, numVehicles);
    avgCBRs = zeros(1, numVehicles);
    avgDataRates = zeros(1, numVehicles);
    
    % Per-vehicle statistics
    for i = 1:numVehicles
        metrics = vehicleMetrics.(vehicleIDs{i});
        if metrics.totalTx > 0
            avgLatencies(i) = mean(metrics.latencies, 'omitnan');
            avgPERs(i) = mean(metrics.per, 'omitnan');
            avgFERs(i) = mean(metrics.fer, 'omitnan');
            avgBERs(i) = mean(metrics.ber, 'omitnan');
            avgCBRs(i) = mean(metrics.cbr, 'omitnan');
            avgDataRates(i) = mean(metrics.dataRates, 'omitnan');
        else
            avgLatencies(i) = NaN;
            avgPERs(i) = NaN;
            avgFERs(i) = NaN;
            avgBERs(i) = NaN;
            avgCBRs(i) = NaN;
            avgDataRates(i) = NaN;
        end
    end

    % Plot Per-Vehicle Statistics
    figure('Name', 'Per-Vehicle Statistics', 'NumberTitle', 'off');
    
    subplot(3, 2, 1);
    bar(avgLatencies);
    title('Average Latency (s) Per Vehicle');
    xlabel('Vehicle ID');
    ylabel('Latency (s)');
    
    subplot(3, 2, 2);
    bar(avgDataRates);
    title('Average Data Rate (Kb/s) Per Vehicle');
    xlabel('Vehicle ID');
    ylabel('Data Rate (Kb/s)');
    
    subplot(3, 2, 3);
    bar(avgPERs);
    title('Packet Error Rate (PER) Per Vehicle');
    xlabel('Vehicle ID');
    ylabel('PER');
    
    subplot(3, 2, 4);
    bar(avgFERs);
    title('Frame Error Rate (FER) Per Vehicle');
    xlabel('Vehicle ID');
    ylabel('FER');
    
    subplot(3, 2, 5);
    bar(avgBERs);
    title('Bit Error Rate (BER) Per Vehicle');
    xlabel('Vehicle ID');
    ylabel('BER');
    
    subplot(3, 2, 6);
    bar(avgCBRs);
    title('Channel Busy Ratio (CBR) Per Vehicle');
    xlabel('Vehicle ID');
    ylabel('CBR');
    
    % Add a graph for MCS
    figure('Name', 'Average MCS Per Vehicle', 'NumberTitle', 'off');
    bar(avgMCS);
    title('Average MCS Per Vehicle');
    xlabel('Vehicle ID');
    ylabel('MCS');

    
    % Overall Statistics
    overallAvgLatency = mean(avgLatencies, 'omitnan');
    overallAvgDataRate = mean(avgDataRates, 'omitnan');
    overallAvgPER = mean(avgPERs, 'omitnan');
    overallAvgFER = mean(avgFERs, 'omitnan');
    overallAvgBER = mean(avgBERs, 'omitnan');
    overallAvgCBR = mean(avgCBRs, 'omitnan');
    
    
    % Display Overall Statistics
    figure('Name', 'Overall Statistics', 'NumberTitle', 'off');
    labels = {'Avg Latency (s)', 'Avg Data Rate (Kb/s)', ...
              'Avg PER', 'Avg FER', 'Avg BER', 'Avg CBR'};
    values = [overallAvgLatency, overallAvgDataRate, ...
              overallAvgPER, overallAvgFER, overallAvgBER, overallAvgCBR];
    
    bar(values);
    set(gca, 'XTickLabel', labels);
    title('Overall Simulation Statistics');
    ylabel('Metric Value');
    xtickangle(45);
end


% Add this function here
function signal = pad_or_truncate_signal(signal, desiredLength)
    % Ensure the signal matches the desired length by padding or truncating
    currentLength = length(signal);
    if currentLength < desiredLength
        signal = [signal; zeros(desiredLength - currentLength, 1)];
    elseif currentLength > desiredLength
        signal = signal(1:desiredLength);
    end
end


function phyOutput = perform_phy_comm(vehID, sourceID, targetID, neighbors, vehicleMetrics, phyParams, simulationParams, cbrParams, customBits)

    % Validate required input parameters
    if nargin < 9
        error('perform_phy_comm: Missing input parameters. Expected 8, received %d.', nargin);
    end
    if isempty(vehID)
        error('perform_phy_comm: Missing vehID (Vehicle ID).');
    end
    if isempty(neighbors)
        error('perform_phy_comm: Neighbors list is empty for vehicle %s.', vehID);
    end
    if ~isfield(vehicleMetrics, vehID)
        error('perform_phy_comm: Vehicle metrics missing for vehicle %s.', vehID);
    end
    if isempty(phyParams)
        error('perform_phy_comm: Missing phyParams structure.');
    end
    if isempty(simulationParams)
        error('perform_phy_comm: Missing simulationParams structure.');
    end
    if isempty(cbrParams)
        error('perform_phy_comm: Missing cbrParams structure.');
    end
    if isempty(customBits)
        error('perform_phy_comm: Missing customBits structure.');
    end
    % Validate that customBits is binary
    if ~all(ismember(customBits, [0, 1]))
        error('perform_phy_comm: customBits must be a binary array.');
    end
    
    % Ensure phyParams.payloadLength exists
    if ~isfield(phyParams, 'payloadLength') || isempty(phyParams.payloadLength)
        error('perform_phy_comm: Missing or empty phyParams.payloadLength.');
    end
    
    % Validate the length of customBits dynamically
    expectedLength = phyParams.payloadLength * 8; % Convert payload length (bytes) to bits
    if length(customBits) > expectedLength
        error('perform_phy_comm: customBits length must be less or equal to %d bits (based on payload length).', expectedLength);
    end

    % Validate required fields in neighbors structure
    requiredNeighborFields = {'id', 'distance', 'P_tx'};
    for i = 1:numel(requiredNeighborFields)
        if ~all(isfield(neighbors, requiredNeighborFields{i}))
            error('perform_phy_comm: Missing field "%s" in neighbors structure.', requiredNeighborFields{i});
        end
    end
    
    % Validate required fields in phyParams structure
    requiredPhyFields = {'transmissionPower', 'bandwidth', 'noiseFigure', 'channelModel', 'MCS', 'frequency'};
    for i = 1:numel(requiredPhyFields)
        if ~isfield(phyParams, requiredPhyFields{i})
            error('perform_phy_comm: Missing field "%s" in phyParams structure.', requiredPhyFields{i});
        end
    end
    
    % Debugging: Output validation results
    disp('[DEBUG] Validation successful for perform_phy_comm.');
    disp('[DEBUG] Neighbors Structure:');
    disp(neighbors);

    % Set random number generator to specific seed
    rand_stream = RandStream('mt19937ar', 'Seed', 0);
    RandStream.setGlobalStream(rand_stream);
    
    %% Simulation Parameters
    % General parameters
    SIM.mcs_vec         = vehicleMetrics.(vehID).MCS;     % Scalar or vector containing MCS values (0...10)
    SIM.snr             = -5:5:35;    % Scalar or vector containing SNR values (dB)
    SIM.ovs             = 1;        % Oversampling factor
    SIM.channel_model   = phyParams.channelModel;        % Channel model (0: AWGN, 1-5: C2C models R-LOS, UA-LOS, C-NLOS, H-LOS and H-NLOS, 6-10: Enhanced C2C models R-LOS-ENH, UA-LOS-ENH, C-NLOS-ENH, H-LOS-ENH and H-NLOS-ENH)
    SIM.use_mex         = false;    % Use MEX functions to accelerate simulation
    SIM.n_iter          = 30;     	% Number of Monte-Carlo iterations
    SIM.max_error       = 100;      % Number of packet errors before moving to next SNR point
    SIM.min_error       = .005;     % Minimum PER target, beyond which, loop moves to next SNR point
    SIM.check_sp        = false;    % Plot Tx spectrum and check for compliance
    SIM.apply_cfo       = false;    % Apply CFO impairment on Tx and Rx
    
    % Transmitter parameters
    TX.payload_len      = 350;      % MPDU_LENGTH / PHY payload length (bytes) (without A-MPDU header and without MAC padding)
    TX.window_en        = false;    % Apply time-domain windowing
    TX.w_beta           = 0;        % Kaiser window beta coefficient for spectral shaping (use "0" for disabling this filter)
    TX.pa_enable        = false;    % Apply PA non-linearity model
    TX.pn_en            = false;    % Model Tx phase noise
    TX.bw_mhz           = 20;       % Bandwidth (10 or 20 MHz)
    
    % Candidate NGV features
    TX.ppdu_fmt         = 2;        % PPDU format (1: Legacy, 2: NGV)
    TX.mid              = 4;        % NGV only: midamble period 4/8/16 symbols
    TX.n_ss             = 1;        % NGV only: Number of MIMO spatial streams (1..2), 2 is not yet supported
    TX.ltf_fmt_init     = 0;        % NGV only: LTF format for MCS 0-9. (0: NGV-LTF-2x (8 us, default), 1: compressed NGV-LTF-1x (4.8 us))
    TX.n_tx_ant         = 1;        % Number of transmit antennas
    
    % Receiver parameters
    RX.n_rx_ant         = 1;        % Number of receive antennas
    RX.bw_mhz           = TX.bw_mhz; % Bandwidth of the received waveform
    RX.pdet_thold_def   = 20;       % Packet detection threshold
    RX.t_depth          = 2;        % Channel tracking time depth averaging (OFDM symbols)
    RX.pn_en            = false;    % Model Rx phase noise
    RX.ldpc_cfg.iter    = 50;       % Number of iterations
    RX.ldpc_cfg.minsum  = 1;        % Only for external decoder 0: Sum-product (gives .5 - 1dB gain) 1: min-sum algorithm
    
    error_char_vec = '.xo!uhnf!'; % For printing receiver message decoding results
    
    ppdu_descriptor = {'p', 'bd'};
    
    % Initialize report structure
    report = struct('P_tx', [], 'MCS', [], 'SNR', [], 'PER', [], 'BER', [], ...
                    'Throughput', [], 'Latency', []);
    
    %% === Optimized SINR Computation with Distance-Weighted Interference ===
    bandwidth = 20e6; % 20 MHz (existing)
    noise_figure = 5; % Noise figure in dB (existing)
    tx_power_dBm = vehicleMetrics.(vehID).transmissionPower; % Existing (dBm)
    TX.P_tx = vehicleMetrics.(vehID).transmissionPower; % Transmit power in dBm
    channel_model = SIM.channel_model; % Existing
    
    % Constants (existing)
    k = 1.38e-23; T = 290; 
    noisePowerLinear = k * T * bandwidth; 
    noisePower_dBm = 10 * log10(noisePowerLinear * 1e3) + noise_figure;
    
    % Convert Tx power to Watts (new)
    tx_power_W = 10^((tx_power_dBm-30)/10); 
    
    % Calculate received power per neighbor (replaces SNR_neighbors)
    P_received = zeros(1, length(neighbors));
    for i = 1:length(neighbors)
        distance = max(neighbors(i).distance, 1); % Avoid div/0
        path_loss = (4*pi*distance/cbrParams.wavelength)^2; % Free-space
        P_received(i) = (tx_power_W * phyParams.G_t * phyParams.G_r) / path_loss;
    end
    
    % Find desired signal (strongest neighbor)
    [P_signal, strongestIdx] = max(P_received);
    
    % Distance-weighted interference (revised)
    P_interference = 0;
    for i = 1:length(neighbors)
        if i ~= strongestIdx
            distance = neighbors(i).distance;
            weight = min(1, (100/distance)^2); % Quadratic attenuation (was ^4)
            P_interference = P_interference + (P_received(i) * weight);
        end
    end
    
    % SINR calculation (existing structure but corrected)
    SINR_linear = P_signal / (P_interference + noisePowerLinear + eps);
    effectiveSnr = 10 * log10(SINR_linear);
    
    % Dynamic neighbor-based adjustment (new)
    num_neighbors = length(neighbors);
    neighbor_adjustment = min(15, 10 + 5*log10(45/max(1,num_neighbors))); 
    effectiveSnr = effectiveSnr + neighbor_adjustment;
    
    % Realistic bounds (revised)
    % effectiveSnr = min(max(effectiveSnr, 5), 30); % Clamp to [5dB, 30dB]
    
    % Store metrics (existing variables)
    vehicleMetrics.(vehID).snr(end + 1) = effectiveSnr; % Store corrected SINR
    SIM.snr = effectiveSnr;
    
    % Debug output (existing)
    fprintf('P_signal = %.2e W, P_interference = %.2e W (%.1f%% of total), Noise = %.2e W\n', ...
        P_signal, P_interference, 100*P_interference/(P_interference+P_signal+noisePowerLinear), noisePowerLinear);
    fprintf('[DEBUG] Vehicle %s | Neighbors: %d | Raw SINR: %.1f dB | Adjusted SINR: %.1f dB\n', ...
            vehID, num_neighbors, 10*log10(SINR_linear), effectiveSnr);
    
    % Throughput logging (unchanged)
    if isfield(vehicleMetrics.(vehID), 'throughput')
        fprintf('→ Throughput: %.2f Mbps\n', mean(vehicleMetrics.(vehID).throughput, 'omitnan')); 
    end

    %% Simulation Execution
    tic;
    
    if (TX.ppdu_fmt == 1 && TX.bw_mhz == 20)
        disp('Warning: non-usual 20 MHz transmissions with 802.11p (4 us symbols, 0.8 us GI, 312.5 kHz subcarrier spacing)');
        TX.n_chan = 1;
        RX.n_chan = 1;
    else
        TX.n_chan = TX.bw_mhz/10;
        RX.n_chan = RX.bw_mhz/10;
    end
    
    % Check if midamble periodicity setting is valid
    if (TX.ppdu_fmt == 1)
        TX.mid = 0;
    else
        if ( ~any(TX.mid == [4 8 16]) )
            error('802.11bd Midamble periodicity (M) should be 4, 8, or 16');
        end
    end
    
    %% Loop for MCS values
    avgTHR = zeros(length(SIM.snr), length(SIM.mcs_vec));
    for i_mcs = 1:length(SIM.mcs_vec)
        % Current MCS value
        TX.mcs = SIM.mcs_vec(i_mcs);
        if (TX.ppdu_fmt == 1  && TX.mcs > 7 ) ...
                || (TX.ppdu_fmt == 2  && TX.mcs == 9 && TX.bw_mhz == 10 )
            % MCS not supported
            continue;
        end
        
        if (TX.ppdu_fmt == 2 && (TX.mcs == 0 || TX.mcs == 10))
            RX.pdet_thold = RX.pdet_thold_def*sqrt(2);
        else
            RX.pdet_thold = RX.pdet_thold_def;
        end
        
        % Initialize channel filter object
        [chan_obj, chan_name] = chan_mod_init(SIM.channel_model, RX.bw_mhz, SIM.ovs, TX.n_tx_ant, RX.n_rx_ant);
        
        % Debugging message
        mcsValue = SIM.mcs_vec;
        PHY0 = tx_phy_params('TX', TX.mcs, TX.payload_len, TX.ppdu_fmt, TX.mid, TX.n_ss, TX.ltf_fmt_init, TX.n_chan, 0, 0);
        fprintf('\nChannel Model %i (%s), %s-MCS %i (%s R=%d/%d), %d MHz, M=%d, %dx%d Antena', SIM.channel_model, chan_name, ppdu_descriptor{TX.ppdu_fmt}, TX.mcs, n_bpscs_to_string(PHY0.n_bpscs),PHY0.r_num,PHY0.r_denom, TX.bw_mhz, TX.mid, TX.n_tx_ant, RX.n_rx_ant);
        phyParams.c_d = mcsToCdMapping(mcsValue); % Update c_d
        packetDuration = TX.payload_len * 8 / (phyParams.bandwidth * phyParams.c_d);

        % Add background traffic contribution to CBR
        % lambda_bg = wblrnd(0.5, 2); % Weibull(k=2, λ=0.5) for background traffic rate
        % packet_size_bg = 50 + (1500-50)*rand(); % Random packet size (50-1500 bytes)
        % mcs_factor = get_MCS_factor(vehicleMetrics.(vehID).MCS); % Spectral efficiency (bits/symbol)
        % symbol_rate = phyParams.bandwidth / 1e6 * 1e6 / 4e-6; % Symbols/sec (4μs duration)
        % packet_duration = (packet_size_bg * 8) / (mcs_factor * symbol_rate); % Channel occupancy time
        % background_cbr = lambda_bg * packet_duration; % Background CBR contribution
        % totalCBR = totalCBR + background_cbr;
        % totalCBR = min(totalCBR, 1); % Ensure CBR ≤ 100%
        % vehicleMetrics.(vehID).cbr(end + 1) = min(totalCBR, 1); % Save CBR, cap at 1

        
        % === Neighbor-Scaled Background Traffic CBR ===

        % === Choose Traffic Profile from simulationParams ===
        if isfield(simulationParams, 'trafficProfile')
            trafficProfile = simulationParams.trafficProfile;
        else
            trafficProfile = 'highway_los'; % Default profile
        end
        
        switch simulationParams.trafficProfile
            case 'awgn_sparse'
                mu = 0.15;
                sigma = 0.02;
                scaling_factor = 10;
        
            case 'rural_los'
                mu = 0.30;
                sigma = 0.03;
                scaling_factor = 15;
        
            case 'highway_los'
                mu = 0.45;
                sigma = 0.04;
                scaling_factor = 17;
        
            case 'highway_nlos'
                mu = 0.60;
                sigma = 0.05;
                scaling_factor = 20;
        
            case 'urban_los'
                mu = 0.70;
                sigma = 0.06;
                scaling_factor = 25;
        
            case 'dense_urban'
                mu = 0.80;
                sigma = 0.05;
                scaling_factor = 30;
        
            case 'max_density'
                mu = 0.90;
                sigma = 0.03;
                scaling_factor = 35;
        
            otherwise
                error('Invalid trafficProfile. Use: awgn_sparse, rural_los, highway_los, highway_nlos, urban_los, dense_urban, or max_density');
        end

        
        neighborList = vehicleMetrics.(vehID).neighbors;
        numNeighbors = length(neighborList);
        total_background_cbr = 0;
        
        if numNeighbors > 0
            neighbor_cbrs = zeros(1, numNeighbors);
            for i = 1:numNeighbors
                sampled = normrnd(mu, sigma);
                sampled = min(max(sampled, 0), 1); % Clamp to [0, 1]
                neighbor_cbrs(i) = sampled / scaling_factor;
            end
            total_background_cbr = min(sum(neighbor_cbrs), 1.0);
        end

        
        % === Packet Config ===
        packet_size_bg = 50 + (1500 - 50) * rand();  % bytes
        mcs_factor = get_MCS_factor(vehicleMetrics.(vehID).MCS);  % bits/symbol
        symbol_rate = phyParams.bandwidth / 4e-6;  % symbols/sec
        packet_duration = (packet_size_bg * 8) / (mcs_factor * symbol_rate);  % sec
        
        lambda_bg = total_background_cbr / packet_duration;
        background_cbr = lambda_bg * packet_duration;  % ≈ total_background_cbr

        
        % Store for logging
        % vehicleMetrics.(vehID).backgroundCBR = [vehicleMetrics.(vehID).backgroundCBR, background_cbr];
        
        switch simulationParams.cbrFormulaType
            case 1
                mcsValue = SIM.mcs_vec;
                r_cs = calculate_r_cs(cbrParams.S, vehicleMetrics.(vehID).transmissionPower, cbrParams.beta, cbrParams.m, cbrParams.wavelength);
                numNeighbors = vehicleMetrics.(vehID).neighbors_number;
                b = vehicleMetrics.(vehID).beaconRate; % Beacon rate (beacons per second)
                p = 10^(vehicleMetrics.(vehID).transmissionPowerBefore / 10) / 1e3; % Current transmission power (Watts)
                if simulationParams.enableRL
                    p_prime = 10^(vehicleMetrics.(vehID).transmissionPower / 10) / 1e3; % Convert dBm to Watts
                else
                    p_prime = p;
                end
                beta = cbrParams.beta; % Path loss exponent
                cbrParams.c_d = mcsToCdMapping(mcsValue); % Update c_d
                C = calculate_C(cbrParams.c_d, cbrParams.b_st, cbrParams.n, cbrParams.t_ps);
                totalCBR = background_cbr + calculate_CBR_type1(r_cs, numNeighbors, b, C, p, p_prime, beta, SIM.mcs_vec, simulationParams);
                disp(['Total CBR: ', num2str(totalCBR)]);
                disp(['Background CBR: ', num2str(background_cbr)]);

            case 2
                mcsValue = SIM.mcs_vec;
                r_cs = calculate_r_cs(cbrParams.S, phyParams.transmissionPower, cbrParams.beta, cbrParams.m, cbrParams.wavelength);
                C = calculate_C(cbrParams.c_d, cbrParams.b_st, cbrParams.n, cbrParams.t_ps);
                totalCBR = calculate_CBR_type2(r_cs, vehicleMetrics.(vehID).neighbors_number, vehicleMetrics.(vehID).beaconRate, C);
                disp(['Total CBR: ', num2str(totalCBR)]);
            case 3
                totalCBR = calculate_CBR_type3(phyParams, vehicleMetrics, cbrParams); % Revised CBR Type 3 logic
           
        end
        

        %% Loop for SNR values
        avgPER = zeros(length(SIM.snr), 1);
        error_char_vec = '.xo!uhnf!'; % decoding outcome symbols
        err_counts = zeros(1, 8);     % counts for each symbol (0–7)

        for i_snr = 1:length(SIM.snr)
            % Total number of transmissions at the current SNR level
            sum_trials = 0;
            % Number of packet errors at the current SNR level
            sum_error = 0;
            
            % Debugging message
            fprintf('\nSNR: %4.1f dB ', SIM.snr(i_snr));
            
            %% Loop for Monte-Carlo iterations Parallel 
            % Initialize parallel pool with maximum available workers
            if isempty(gcp('nocreate'))
                poolObj = parcluster('local');
                maxWorkers = poolObj.NumWorkers;
                parpool('local', maxWorkers);
            end
            
            % Initialize results container
            sum_error = 0;
            sum_trials = 0;
            err_counts = zeros(1, 8);
            
            % Preallocate storage for iteration-level results
            sum_error_vec = zeros(1, SIM.n_iter);
            sum_trials_vec = zeros(1, SIM.n_iter);
            err_counts_mat = zeros(SIM.n_iter, 8);
            
            % Preallocate storage for data_f_mtx (optional)
            data_f_mtx_cell = cell(1, SIM.n_iter);
            
            % Read-only variables
            TX_par = TX;
            RX_par = RX;
            SIM_par = SIM;
            phyParams_par = phyParams;
            neighbors_par = neighbors;
            customBits_par = customBits;
            
            parfor i_iter = 1:SIM_par.n_iter
                sum_error_iter = 0;
                sum_trials_iter = 0;
                err_counts_iter = zeros(1, 8);
            
                rand_stream_iter = RandStream('mt19937ar', 'Seed', i_iter);
            
                custom_bits = customBits_par;
                if length(custom_bits) < TX_par.payload_len * 8
                    custom_bits = [custom_bits, zeros(1, TX_par.payload_len * 8 - length(custom_bits))];
                elseif length(custom_bits) > TX_par.payload_len * 8
                    custom_bits = custom_bits(1:TX_par.payload_len * 8);
                end
            
                if SIM_par.use_mex
                    [tx_wf, data_f_mtx_iter, ~, PHY_par] = sim_tx_mex(TX_par);
                else
                    [tx_wf, data_f_mtx_iter, ~, PHY_par] = sim_tx(TX_par);
                end
            
                data_f_mtx_cell{i_iter} = data_f_mtx_iter;
            
                if SIM_par.apply_cfo
                    cfo_err = sum(rand(rand_stream_iter, 2, 1) - 0.5) * 10e-6;
                    tx_wf = apply_cfo(tx_wf, cfo_err);
                end
            
                tx_wf = add_tx_pn(tx_wf, TX_par.pn_en);
            
                if (RX_par.bw_mhz == 20 && TX_par.bw_mhz == 10)
                    [tx_wf, ~] = upsample_tx(tx_wf, 2);
                    tx_wf = tx_wf .* exp(-1j * 2 * pi * (1:length(tx_wf)) / 128 * 32).';
                end
            
                [tx_wf, ovs_filt_len] = upsample_tx(tx_wf, SIM_par.ovs);
                tx_wf = pa_model(tx_wf, TX_par.pa_enable);
            
                s0_len = randi(rand_stream_iter, [100, 200]);
                tx_wf_full = [zeros(s0_len * SIM_par.ovs, TX_par.n_tx_ant); tx_wf; zeros((400 - s0_len) * SIM_par.ovs, TX_par.n_tx_ant)];
            
                if (SIM_par.channel_model == 0)
                    rx_wf = repmat(tx_wf_full, 1, RX_par.n_rx_ant);
                else
                    chan_obj = chan_mod_init(SIM_par.channel_model, RX_par.bw_mhz, SIM_par.ovs, TX_par.n_tx_ant, RX_par.n_rx_ant);
                    reset(chan_obj);
                    rx_wf = step(chan_obj, tx_wf_full);
                end
            
                rx_wf = downsample_rx(rx_wf, SIM_par.ovs, ovs_filt_len);
                rx_wf = add_rx_pn(rx_wf, RX_par.pn_en);
            
                if abs(neighbors_par(1).dopplerShift) > 0
                    Ts = 1 / phyParams_par.bandwidth;
                    t = (0:length(rx_wf) - 1)' * Ts;
                    rx_wf = rx_wf .* exp(1j * 2 * pi * neighbors_par(1).dopplerShift * t);
                end
            
                rx_wf = awgn(rx_wf, SIM_par.snr(i_snr), 'measured', rand_stream_iter);
            
                if SIM_par.use_mex
                    err = sim_rx_mex(rx_wf, data_f_mtx_iter, RX_par, TX_par, s0_len);
                else
                    err = sim_rx(rx_wf, data_f_mtx_iter, RX_par, TX_par, s0_len);
                end
            
                sum_error_iter = sum_error_iter + (err > 0);
                sum_trials_iter = sum_trials_iter + 1;
                if err >= 0 && err < 8
                    err_counts_iter(err + 1) = err_counts_iter(err + 1) + 1;
                end
            
                sum_error_vec(i_iter) = sum_error_iter;
                sum_trials_vec(i_iter) = sum_trials_iter;
                err_counts_mat(i_iter, :) = err_counts_iter;
            end
            
            sum_error = sum(sum_error_vec);
            sum_trials = sum(sum_trials_vec);
            err_counts = sum(err_counts_mat, 1);
            
            data_f_mtx = data_f_mtx_cell{1};



            
            
            
            %% === Enhanced PER, BER, SER, Throughput and Latency Calculation ===
            
            % Old formula
            %{
             avgPER(i_snr) = sum_error/sum_trials;
            % Calculate BER
            BER = sum_error / (sum_trials * TX.payload_len * 8); % Bit Error Rate
            fprintf('Bit Error Rate (BER): %.4f\n', BER);
            
            % Calculate SER
            mcsTable = [
                    0, 2, 1/2;   % MCS 0  -> BPSK (M = 2), Code Rate = 1/2
                    1, 4, 1/2;   % MCS 1  -> QPSK (M = 4), Code Rate = 1/2
                    2, 4, 3/4;   % MCS 2  -> QPSK (M = 4), Code Rate = 3/4
                    3, 16, 1/2;  % MCS 3  -> 16-QAM (M = 16), Code Rate = 1/2
                    4, 16, 3/4;  % MCS 4  -> 16-QAM (M = 16), Code Rate = 3/4
                    5, 64, 2/3;  % MCS 5  -> 64-QAM (M = 64), Code Rate = 2/3
                    6, 64, 3/4;  % MCS 6  -> 64-QAM (M = 64), Code Rate = 3/4
                    7, 64, 5/6;  % MCS 7  -> 64-QAM (M = 64), Code Rate = 5/6
                    8, 256, 3/4; % MCS 8  -> 256-QAM (M = 256), Code Rate = 3/4
                    9, 256, 5/6; % MCS 9  -> 256-QAM (M = 256), Code Rate = 5/6
                    10, 1024, 3/4;% MCS 10 -> 1024-QAM (M = 1024), Code Rate = 3/4
                ];
          
            modulationOrder = mcsTable(phyParams.MCS + 1, 2); % Get modulation order from MCS
            ser = 1 - (1 - BER)^(1 / log2(modulationOrder)); % Approximate SER from BER
            
            % Correct throughput calculation
            symbol_time = 4e-6; % OFDM symbol duration in seconds (20 MHz bandwidth)
            total_symbols = TX.payload_len * 8 / PHY0.n_dbps;
            frame_time = total_symbols * symbol_time; % Total frame time in seconds
            % throughput = (TX.payload_len * 8 * (1 - avgPER(i_snr))) / frame_time / 1e6; % Mbps
            throughput = (TX.payload_len * 8 * (1 - avgPER(i_snr))) / (frame_time * (1 + avgPER(i_snr))) / 1e6;
            fprintf('Calculated Throughput: %.4f Mbps\n', throughput);
            
            % Calculate latency
            % calc_latency = (TX.payload_len * 8) / (PHY0.n_sd * PHY0.n_dbps / PHY0.n_cbps * PHY0.n_bpscs / 8); % µs
            calc_latency = (TX.payload_len * 8) / ((PHY0.n_sd * PHY0.n_dbps) / (PHY0.n_cbps * PHY0.n_bpscs / 8));
            fprintf('Calculated Latency: %.4f µs\n', calc_latency);
            
            %}

           
            %% === Enhanced PER, BER, SER, Throughput and Latency Calculation ===
            % Revised formulas with proper scaling and realistic assumptions
            
            % Packet Error Rate - now includes neighbor interference effects
            neighbor_penalty = min(0.3, 0.05 + 0.01*numel(vehicleMetrics.(vehID).neighbors)); % 5% base + 1% per neighbor
            avgPER(i_snr) = min(0.99, (sum_error / sum_trials) + neighbor_penalty - (sum_error / sum_trials)*neighbor_penalty);
            PER = avgPER(i_snr);
            
            % Debugging output
            fprintf('[ITER-LOG] i_snr=%d | sum_error=%d | sum_trials=%d | PER=%.4f | Neighbors=%d\n', ...
                    i_snr, sum_error, sum_trials, PER, numel(vehicleMetrics.(vehID).neighbors));
            
            % MCS to modulation mapping
            mcsTable = [
                0, 2, 1/2;   % MCS 0: BPSK, R=1/2
                1, 4, 1/2;   % MCS 1: QPSK, R=1/2
                2, 4, 3/4;   % MCS 2: QPSK, R=3/4
                3, 16, 1/2;  % MCS 3: 16QAM, R=1/2
                4, 16, 3/4;  % MCS 4: 16QAM, R=3/4
                5, 64, 2/3;  % MCS 5: 64QAM, R=2/3
                6, 64, 3/4;  % MCS 6: 64QAM, R=3/4
                7, 64, 5/6;  % MCS 7: 64QAM, R=5/6
                8, 256, 3/4; % MCS 8: 256QAM, R=3/4
                9, 256, 5/6; % MCS 9: 256QAM, R=5/6
                10,1024,3/4; % MCS 10:1024QAM,R=3/4
            ];
            
            % Get current MCS parameters
            modulationOrder = mcsTable(PHY0.mcs + 1, 2);
            codeRate = mcsTable(PHY0.mcs + 1, 3);
            snr_linear = 10^(effectiveSnr / 10);
            
            % Calculate theoretical BER based on modulation and coding
            switch modulationOrder
                case 2 % BPSK
                    ber_theoretical = qfunc(sqrt(2 * snr_linear));
                case 4 % QPSK
                    ber_theoretical = qfunc(sqrt(snr_linear));
                otherwise % M-QAM
                    ber_theoretical = (4/log2(modulationOrder)) * (1-1/sqrt(modulationOrder)) * ...
                                     qfunc(sqrt(3*snr_linear/(modulationOrder-1)));
            end
            
            % Adjust for coding gain (simplified approximation)
            coding_gain = 10*log10(codeRate);
            effective_snr_linear = snr_linear * 10^(coding_gain/10);
            BER = ber_theoretical * (1 + 0.2*randn()); % Add small randomness for realism
            
            % Calculate SER based on BER (more accurate relationship)
            ser = 1 - (1 - BER)^log2(modulationOrder);
            
            % Calculate actual bit errors from PER and BER
            if PER > 0
                avg_bit_errors_per_packet = BER * TX.payload_len * 8;
            else
                avg_bit_errors_per_packet = 0;
            end
            
            % Throughput calculation with overhead considerations
            symbol_time = 4e-6; % OFDM symbol duration
            bits_per_symbol = PHY0.n_dbps; % Data bits per symbol
            total_bits = TX.payload_len * 8;
            total_symbols_needed = ceil(total_bits / bits_per_symbol);
            
            % Include PHY and MAC overhead
            phy_overhead = 20e-6; % Preamble + header
            mac_overhead = 34*8 / (PHY0.n_dbps / symbol_time); % MAC header transmission time
            
            % Total frame time including overhead
            frame_time = phy_overhead + mac_overhead + (total_symbols_needed * symbol_time);
            
            % Goodput calculation (successful payload bits / total time)
            successfulBits = (sum_trials - sum_error) * total_bits;
            totalTime = sum_trials * frame_time;
            
            if totalTime > 0
                throughput = successfulBits / totalTime / 1e6; % Mbps
            else
                throughput = 0;
            end
            
            % Latency calculation with realistic retransmission model
            phy_latency = frame_time; % Single transmission time
            
            % MAC layer timing parameters (802.11bd)
            DIFS = 34e-6;
            SLOT_TIME = 13e-6;
            CWmin = 15;
            avg_backoff = (CWmin / 2) * SLOT_TIME;
            
            % Retry model - exponential backoff
            maxRetries = 3;
            retry_probability = min(PER * 1.2, 0.95); % Slightly higher than PER for realism
            
            % Calculate expected number of transmissions
            if PER > 0
                expected_transmissions = min((1 / (1 - retry_probability)), maxRetries + 1);
            else
                expected_transmissions = 1;
            end
            
            % Total latency including retries and backoffs
            calc_latency = expected_transmissions * phy_latency + ...
                           (expected_transmissions - 1) * (DIFS + avg_backoff);
            
            % Adjust for high BER conditions
            if BER > 0.001
                calc_latency = calc_latency * (1 + min(BER * 5, 2)); % Up to 3x latency for very high BER
            end
            
            % Debug outputs
            fprintf('Revised Calculations:\n');
            fprintf('BER: %.2e, SER: %.2e, Throughput: %.2f Mbps, Latency: %.4f ms\n', ...
                    BER, ser, throughput, calc_latency*1000);
            fprintf('Expected transmissions: %.2f, PHY latency: %.2f μs\n', ...
                    expected_transmissions, phy_latency*1e6);
            
            % Sanity check on MCS vs SNR
            required_snr = 10 + (PHY0.mcs * 2); % Approximate required SNR for MCS
            if effectiveSnr < required_snr - 3
                warning('SINR %.2f dB too low for MCS %d (needs ~%.1f dB)', ...
                        effectiveSnr, PHY0.mcs, required_snr);
            end


            % Simpan hasil ke vehicleMetrics
            if isfield(vehicleMetrics, vehID)
                vehicleMetrics.(vehID).ber(end + 1) = BER;
                vehicleMetrics.(vehID).per(end + 1) = avgPER(i_snr);
                vehicleMetrics.(vehID).throughput(end + 1) = throughput;
                vehicleMetrics.(vehID).dataRates = throughput;
                vehicleMetrics.(vehID).latencies(end + 1) = calc_latency;
                vehicleMetrics.(vehID).ser = [vehicleMetrics.(vehID).ser, ser];
                % vehicleMetrics.(vehID).pdr = vehicleMetrics.(vehID).successfulTx / max(1, vehicleMetrics.(vehID).totalTx);
                vehicleMetrics.(vehID).pdr(end+1) = 1 - avgPER(i_snr);  % From PHY
            else
                warning('[WARNING] Vehicle ID %s not found in vehicleMetrics. Skipping data storage.', vehID);
            end

            % Store results in the report
            report.P_tx = [report.P_tx; TX.P_tx];
            report.MCS = [report.MCS; TX.mcs];
            report.SNR = [report.SNR; SIM.snr(i_snr)];
            report.PER = [report.PER; avgPER(i_snr)];
            report.BER = [report.BER; BER];
            report.Throughput = [report.Throughput; throughput];
            report.Latency = [report.Latency; calc_latency];
    
            % If PER drops below min_error, break SNR loop
            if (sum_error/sum_trials < SIM.min_error)
                break;
            end
        end
        err_types = ["OK", "PacketLoss", "SyncFail", "HeaderFail", ...
             "Unsupported", "ChanEstFail", "NoiseFail", "FrameFail"];
        fprintf('[ERROR-DETAILS] SNR: %.1f dB\n', SIM.snr(i_snr));
        for ei = 1:numel(err_types)
            fprintf('  %-12s: %d\n', err_types(ei), err_counts(ei));
        end
        fprintf('\n');
        
        % Find throughput efficiency factor (affected by midamble periodicity and duration of the midamble symbols, which is 8, 4.8 or 14.4 us)
        if (TX.mid == 0)
            eff = 1;
        else
            eff = (TX.mid*8)/(TX.mid*8 + PHY0.n_ss * PHY0.t_ngvltf);
        end
        % Goodput calculation
        drate = eff*PHY0.n_sd*PHY0.n_dbps./PHY0.n_cbps.*PHY0.n_bpscs/8e-6*1e-6;
        avgTHR(:, i_mcs) = (1 - avgPER).*repmat(drate, size(avgPER, 1), 1);
    end
    
    toc;
    
    % Generate table report
    table_report = table(report.P_tx, report.MCS, report.SNR, report.PER, ...
                         report.BER, report.Throughput, report.Latency, ...
                         'VariableNames', {'P_tx', 'MCS', 'SNR', 'PER', 'BER', 'Throughput', 'Latency'});
    disp(table_report);

    % At the end of the function, aggregate these metrics:
    % success = (sum_error == 0); % Successful transmission if no errors
    latency = calc_latency; % Latency from simulation
    dataRate = throughput; % Data rate calculated
    errorMetrics = struct('PER', avgPER, 'BER', BER, 'SER', ser); % Error metrics including SER
    debugInfo = table_report; % Debugging details

    % Construct the output structure with the computed metrics
    phyOutput = struct( ...
        'latency', latency, ...
        'dataRate', dataRate, ...
        'errorMetrics', errorMetrics, ...
        'debugInfo', debugInfo, ...
        'effectiveSnr', effectiveSnr, ...
        'totalCBR', min(totalCBR, 1), ...
        'PER', avgPER, ...
        'BER', BER, ...
        'SER', ser, ...
        'receivedBits', data_f_mtx ...
    );

    return;
end


% =========================================================================
% Visualize Vehicles Movement
% =========================================================================


function visualize_vehicle_movement_with_communication(fcdData, commRange, beaconLog, timeStep, simulationDuration)
    % VISUALIZE_VEHICLE_MOVEMENT_WITH_COMMUNICATION
    % Visualizes vehicle movements and communication links based on FCD data.
    %
    % Inputs:
    %   fcdData - Vehicle mobility data loaded from SUMO's FCD file.
    %   commRange - Communication range of vehicles (in meters).
    %   beaconLog - Communication log (source, target, timestamp).
    %   timeStep - Time step for updating visualization.
    %   simulationDuration - Total duration of the simulation (in seconds).

    % Get unique time points
    timePoints = unique([fcdData.time]);

    % Prepare the figure
    figure('Name', 'Vehicle Movement and Communication', 'NumberTitle', 'off');
    axis equal;
    hold on;
    grid on;
    xlabel('X Coordinate (m)');
    ylabel('Y Coordinate (m)');
    title('Vehicle Movement and Communication Visualization');

    % Main visualization loop
    for tIdx = 1:length(timePoints)
        % Get current time
        currentTime = timePoints(tIdx);

        % Filter vehicles at the current time
        currentFrame = fcdData([fcdData.time] == currentTime);

        % Clear the axes for the current frame
        cla;

        % Plot all vehicles
        neighbors = [];
        for i = 1:length(currentFrame)
            vehID = currentFrame(i).id;
            x = currentFrame(i).x;
            y = currentFrame(i).y;

            % Plot vehicle as a point
            plot(x, y, 'bo', 'MarkerSize', 8, 'LineWidth', 2);
            text(x + 2, y, vehID, 'FontSize', 8, 'Color', 'blue');

            % Plot communication range as a circle
            rectangle('Position', [x - commRange, y - commRange, 2 * commRange, 2 * commRange], ...
                      'Curvature', [1, 1], 'EdgeColor', 'cyan', 'LineStyle', '--');
        end

        if isempty(neighbors)
            warning('[WARNING] No neighbors found for vehicle %s at time %.2f.', vehID, currentTime);
            vehicleMetrics.(vehID).snr(end + 1) = NaN;
            continue; % skip ALL transmission
        end



        % Plot communication links based on the beacon log
        currentLogs = beaconLog([beaconLog{:, 3}] == currentTime, :);
        for logIdx = 1:size(currentLogs, 1)
            sourceID = currentLogs{logIdx, 1};
            targetID = currentLogs{logIdx, 2};

            % Get positions of source and target vehicles
            sourceVehicle = currentFrame(strcmp({currentFrame.id}, sourceID));
            targetVehicle = currentFrame(strcmp({currentFrame.id}, targetID));

            if ~isempty(sourceVehicle) && ~isempty(targetVehicle)
                % Draw a line representing communication
                line([sourceVehicle.x, targetVehicle.x], [sourceVehicle.y, targetVehicle.y], ...
                     'Color', 'red', 'LineWidth', 1.5, 'LineStyle', '-');
            end
        end

        % Pause to create an animation effect
        pause(timeStep);
    end

    disp('[INFO] Visualization completed successfully.');
end


% =========================================================================
% RL Communication Function
% =========================================================================
function rlResponse = communicate_with_rl(rlClient, allRLData)
    % Initialize response structure
    rlResponse = struct();
    
    % Parameters
    maxVehiclesPerMessage = 20;  % Prevent oversized messages
    vehicleIDs = fieldnames(allRLData);
    numMessages = ceil(numel(vehicleIDs)/maxVehiclesPerMessage);
    
    for msgIdx = 1:numMessages
        % Select subset of vehicles
        startIdx = (msgIdx-1)*maxVehiclesPerMessage + 1;
        endIdx = min(msgIdx*maxVehiclesPerMessage, numel(vehicleIDs));
        currentIDs = vehicleIDs(startIdx:endIdx);
        
        % Create message data
        messageData = struct();
        for v = 1:numel(currentIDs)
            vehID = currentIDs{v};
            currentData = allRLData.(vehID);
            
            % Ensure all required fields exist with valid values
            if ~isfield(currentData, 'CBR') || isnan(currentData.CBR)
                currentData.CBR = 0;
            end
            if ~isfield(currentData, 'SNR') || isnan(currentData.SNR)
                currentData.SNR = 0;
            end
            if ~isfield(currentData, 'neighbors') || isempty(currentData.neighbors)
                currentData.neighbors = 0;
            end
            % Add other field validations as needed
            
            messageData.(vehID) = currentData;
        end
        
        % Convert to JSON
        try
            rlDataJSON = jsonencode(messageData, 'ConvertInfAndNaN', false);
            messageBytes = unicode2native(rlDataJSON, 'UTF-8');
        catch
            warning('JSON encoding failed');
            continue;
        end
        
        % Check/create connection
        if isempty(rlClient) || ~isvalid(rlClient)
            try
                rlClient = tcpclient('127.0.0.1', 5000, 'Timeout', 30);
                configureTerminator(rlClient, 'LF');  % Ensure consistent line endings
            catch
                warning('Connection failed');
                return;
            end
        end
        
        % Send message with length header
        try
            % Send message length (4 bytes, little-endian)
            msgLength = uint32(length(messageBytes));
            write(rlClient, typecast(msgLength, 'uint8'));
            
            % Send message content
            write(rlClient, messageBytes);
        catch
            warning('Message send failed');
            return;
        end
        
        % Receive response
        try
            % Read response header (4 bytes)
            startTime = tic;
            while rlClient.BytesAvailable < 4 && toc(startTime) < 10
                pause(0.01);
            end
            
            if rlClient.BytesAvailable < 4
                warning('No response header received');
                return;
            end
            
            responseLength = typecast(read(rlClient, 4), 'uint32');
            
            % Read response body
            responseData = uint8([]);
            while length(responseData) < responseLength && toc(startTime) < 10
                if rlClient.BytesAvailable > 0
                    responseData = [responseData; read(rlClient, rlClient.BytesAvailable)];
                end
                pause(0.01);
            end
            
            if length(responseData) < responseLength
                warning('Incomplete response received');
                return;
            end
            
            % Parse response
            try
                responseJSON = native2unicode(responseData, 'UTF-8');
                partialResponse = jsondecode(responseJSON);
                
                % Merge responses
                if isempty(fieldnames(rlResponse))
                    rlResponse = partialResponse;
                else
                    respVehicles = fieldnames(partialResponse.vehicles);
                    for v = 1:numel(respVehicles)
                        vehID = respVehicles{v};
                        rlResponse.vehicles.(vehID) = partialResponse.vehicles.(vehID);
                    end
                end
            catch
                warning('Response parsing failed');
            end
        catch
            warning('Response receive failed');
        end
    end
end


% =========================================================================
% Calculate SNR From Neighbors
% =========================================================================


function SNR = calculate_snr_from_neighbors(neighbors, bandwidth, noise_figure, channel_model, TX_P_tx, phyParams, varargin)
    % calculate_snr_from_neighbors: Calculates SNR with reduced neighbor influence
    % Enhanced features:
    % 1. Distance-weighted interference scaling
    % 2. Adaptive fading correlation for dense networks
    % 3. Noise-floor aware interference rejection

    % Constants
    k = 1.38e-23; T = 290; % Boltzmann
    P_thermal_noise_dBm = 10 * log10(k * T * bandwidth * 1e3);
    P_noise_dBm = P_thermal_noise_dBm + noise_figure;
    noise_floor_linear = 10^(P_noise_dBm/10)/1000; % Convert to Watts

    % Antenna and environment
    G_t = phyParams.G_t; G_r = phyParams.G_r;
    frequency_hz = phyParams.frequency;
    
    % Initialize
    num_neighbors = length(neighbors);
    SNR = zeros(1, num_neighbors + 1); % +1 for self-node
    
    % Pre-compute distance scaling factors
    distance_weights = zeros(1, num_neighbors);
    for i = 1:num_neighbors
        distance = max(neighbors(i).distance, 1e-3);
        % Distance-based scaling (cubic attenuation beyond 50m)
        if distance <= 50
            distance_weights(i) = 1;
        else
            distance_weights(i) = (50/distance)^3; % Strong distance discount
        end
    end

    % Calculate fading with correlation for nearby nodes
    fading_cache = containers.Map();
    for i = 1:num_neighbors
        distance = max(neighbors(i).distance, 1e-3);
        
        % Get fading based on channel model
        switch channel_model
            case 0
                fading = 0;
            case 1
                if distance < 30
                    fading = get_correlated_fading(distance, fading_cache, 0.1, 0);
                else
                    fading = 10 * log10(abs(raylrnd(0.1)));
                end
            case 2
                if distance < 30
                    fading = get_correlated_fading(distance, fading_cache, 0.5, 0);
                else
                    fading = 10 * log10(abs(raylrnd(0.5)));
                end
            case 3
                if distance < 30
                    fading = get_correlated_fading(distance, fading_cache, 1, 0);
                else
                    fading = 10 * log10(abs(raylrnd(1)));
                end
            case 4
                if distance < 30
                    K = 10;
                    sigma = sqrt(1/(10^(K/10)+1));
                    mu = sqrt(10^(K/10)/(2*(10^(K/10)+1)));
                    fading = 10*log10(abs(mu + sigma*randn + 1i*(mu + sigma*randn)));
                else
                    fading = 10*log10(abs(raylrnd(0.2))); % Reduced variance
                end
            case 5
                if distance < 30
                    fading = get_correlated_fading(distance, fading_cache, 0.8, 0);
                else
                    fading = 10 * log10(abs(raylrnd(0.8)));
                end
            case 6
                if distance < 30
                    fading = get_correlated_fading(distance, fading_cache, 0.1, 0.5);
                else
                    fading = 10 * log10(abs(raylrnd(0.1))) + normrnd(0, 0.3); % Reduced shadowing
                end
            case 7
                if distance < 30
                    fading = get_correlated_fading(distance, fading_cache, 0.5, 1);
                else
                    fading = 10 * log10(abs(raylrnd(0.5))) + normrnd(0, 0.7); % Reduced shadowing
                end
            case 8
                if distance < 30
                    fading = get_correlated_fading(distance, fading_cache, 1, 2);
                else
                    fading = 10 * log10(abs(raylrnd(1))) + normrnd(0, 1.5); % Reduced shadowing
                end
            case 9
                if distance < 30
                    fading = get_correlated_fading(distance, fading_cache, 0.2, 0.5);
                else
                    fading = 10 * log10(abs(raylrnd(0.2))) + normrnd(0, 0.3); % Reduced shadowing
                end
            case 10
                if distance < 30
                    fading = get_correlated_fading(distance, fading_cache, 0.8, 1);
                else
                    fading = 10 * log10(abs(raylrnd(0.8))) + normrnd(0, 0.7); % Reduced shadowing
                end
            otherwise
                error('Unsupported channel model: %d', channel_model);
        end

        % Apply distance-weighted path loss
        distance_km = distance/1000;
        if distance_km <= 0.01
            path_loss = 22.7*log10(distance) + 27.0 + 20*log10(frequency_hz/5e9);
        else
            path_loss = 40.0*log10(distance) + 7.56 - 17.3*log10(1.6) ...
                      - 17.3*log10(3.0) + 2.7*log10(frequency_hz/5e9);
        end

        % Apply distance weighting to received power
        P_rx = neighbors(i).P_tx + G_t + G_r - path_loss - fading;
        P_rx_linear = (10^(P_rx/10)/1000 * distance_weights(i)); % Linear-domain discount
        
        % Only count significant interferers
        if P_rx_linear > 10*noise_floor_linear
            SNR(i) = 10*log10(P_rx_linear*1000) - P_noise_dBm;
        else
            SNR(i) = -Inf; % Discard
        end
    end

    % Self-node calculation (unchanged)
    if nargin >= 5 && ~isempty(TX_P_tx)
        own_distance = 1;
        path_loss_own = 20 * log10(own_distance) + 20 * log10(frequency_hz) - 147.55;
        P_rx_own = TX_P_tx + G_t + G_r - path_loss_own;
        SNR(end) = P_rx_own - P_noise_dBm;
    else
        SNR(end) = NaN;
    end
    
    % Remove discarded interferers
    SNR(SNR == -Inf) = [];
end

function fading = get_correlated_fading(distance, cache, rayleigh_param, shadowing_std)
    % Helper function for correlated fading
    key = sprintf('%.1f_%.2f_%.2f', distance, rayleigh_param, shadowing_std);
    if ~isKey(cache, key)
        if shadowing_std > 0
            cache(key) = 10*log10(abs(raylrnd(rayleigh_param))) + normrnd(0, shadowing_std*0.7);
        else
            cache(key) = 10*log10(abs(raylrnd(rayleigh_param)));
        end
    end
    fading = cache(key);
end

% =========================================================================
% Calculate Communication Range and Receive Power
% =========================================================================

function R = calculate_comm_range(P_t, G_t, G_r, f, R_s, alpha)
    % Calculate communication range using the given formula
    % Inputs:
    % P_t: Transmission power (W)
    % G_t: Transmitter antenna gain
    % G_r: Receiver antenna gain
    % f: Frequency (Hz)
    % R_s: Receiver sensitivity (W)
    % alpha: Path loss exponent
    % Output:
    % R: Communication range (meters)

    % Calculate communication range (meters) using the provided formula
    c = 3e8; % Speed of light (m/s)
    lambda = c / f; % Wavelength (m)
    R = ((P_t * G_t * G_r * lambda^2) / ((4 * pi)^2 * R_s))^(1 / alpha);

end

function P_r = calculate_received_power(P_t, G_t, G_r, distance, alpha, f)
    % Calculate received power (dBm) using path loss and fading
    c = 3e8; % Speed of light (m/s)
    lambda = c / f; % Wavelength (m)
    pathLoss = (4 * pi * distance / lambda)^alpha; % Path loss (linear scale)
    pathLoss_dB = 10 * log10(pathLoss); % Path loss (dB)
    P_r = P_t + G_t + G_r - pathLoss_dB; % Received power (dBm)
end


% =========================================================================
% Resolve Duplicate Position
% =========================================================================

function fcdData = resolve_duplicate_positions(fcdData)
    % Function to adjust positions of vehicles with identical coordinates
    % Inputs:
    %   fcdData - Array of structures with fields: id, time, x, y
    % Outputs:
    %   fcdData - Updated FCD data with resolved duplicates

    uniqueTimes = unique([fcdData.time]);
    perturbationStep = 0.01; % Small step to separate positions

    for t = uniqueTimes
        % Extract all vehicles at the same timestamp
        currentFrame = fcdData([fcdData.time] == t);
        positions = [currentFrame.x; currentFrame.y]';
        
        % Find duplicate positions
        [uniquePositions, ~, idx] = unique(positions, 'rows', 'stable');
        counts = accumarray(idx, 1);
        duplicates = uniquePositions(counts > 1, :);
        
        for d = 1:size(duplicates, 1)
            % Get indices of all vehicles with the same position
            duplicateIdx = find(positions(:, 1) == duplicates(d, 1) & ...
                                positions(:, 2) == duplicates(d, 2));
            
            % Perturb the positions of all but the first vehicle
            for i = 2:length(duplicateIdx)
                fcdData(duplicateIdx(i)).x = fcdData(duplicateIdx(i)).x + perturbationStep;
                fcdData(duplicateIdx(i)).y = fcdData(duplicateIdx(i)).y + perturbationStep;
                perturbationStep = perturbationStep + 0.01; % Increment for next adjustment
            end
        end
    end
end


function save_neighbor_data(fcdData, commRange, outputFile, phyParams)
    % Save neighbors with communication range and received power to file

    % Initialize table for saving results
    resultTable = table('Size', [0, 5], ...
                        'VariableTypes', {'double', 'string', 'string', 'double', 'double'}, ...
                        'VariableNames', {'Timestamp', 'VehicleID', 'Neighbors', 'CommRange', 'P_r'});

    % Process each timestamp
    uniqueTimes = unique([fcdData.time]);
    for t = uniqueTimes
        % Filter vehicles at the current timestamp
        currentFrame = fcdData([fcdData.time] == t);

        % Extract positions and IDs
        numVehicles = length(currentFrame);
        positions = [[currentFrame.x]', [currentFrame.y]'];
        vehicleIDs = {currentFrame.id};

        % Compute pairwise distances
        distances = squareform(pdist(positions)); % Pairwise distances matrix

        % Identify neighbors for each vehicle
        for i = 1:numVehicles
            vehID = vehicleIDs{i};
            neighbors = {};
            for j = 1:numVehicles
                if i ~= j && distances(i, j) <= commRange
                    neighbors{end + 1} = vehicleIDs{j}; %#ok<AGROW>

                    % Calculate communication range and received power
                    P_t = phyParams.transmissionPower; % Transmission power (dBm)
                    G_t = 1; % Transmitter gain
                    G_r = 1; % Receiver gain
                    f = phyParams.frequency; % Frequency (Hz)
                    R_s = 10^(-90 / 10); % Receiver sensitivity (Watts)
                    alpha = 2.5; % Path loss exponent

                    % Calculate values
                    commRangeCalc = calculate_comm_range(10^(P_t / 10) / 1e3, G_t, G_r, f, R_s, alpha);
                    P_r = calculate_received_power(P_t, G_t, G_r, distances(i, j), alpha, f);

                    % Save to result table
                    resultTable = [resultTable; {t, vehID, strjoin(neighbors, ', '), commRangeCalc, P_r}]; %#ok<AGROW>
                end
            end
        end
    end

    % Write result table to file
    if endsWith(outputFile, '.csv')
        writetable(resultTable, outputFile);
    elseif endsWith(outputFile, {'.xls', '.xlsx'})
        writetable(resultTable, outputFile, 'FileType', 'spreadsheet');
    else
        error('Unsupported file format. Use .csv or .xls/.xlsx.');
    end

    disp(['[INFO] Results saved to: ', outputFile]);
end



% =========================================================================
% LDPC Matrix Generation
% =========================================================================

function H = generateLDPCMatrix(codeRate)
    % Define basic parity-check matrix sizes based on standard LDPC sizes
    baseMatrixSize = 64;  % Default size (can be adjusted)

    % Adjust matrix size based on code rate
    switch codeRate
        case 1/2
            H = randi([0, 1], baseMatrixSize, 2 * baseMatrixSize); % 1/2 rate LDPC
        case 2/3
            H = randi([0, 1], baseMatrixSize, (3/2) * baseMatrixSize); % 2/3 rate
        case 3/4
            H = randi([0, 1], baseMatrixSize, (4/3) * baseMatrixSize); % 3/4 rate
        case 5/6
            H = randi([0, 1], baseMatrixSize, (6/5) * baseMatrixSize); % 5/6 rate
        otherwise
            error('Unsupported code rate.');
    end

    % Ensure matrix is sparse to optimize LDPC decoding
    H = sparse(H);

    % Display generated matrix size
    fprintf('[INFO] Generated LDPC Parity-Check Matrix of size %dx%d for Code Rate %.2f\n', ...
            size(H, 1), size(H, 2), codeRate);
end

% =========================================================================
% Save Simulation Results with Validation Logging
% =========================================================================
function save_simulation_results(vehicleMetrics, timestamps, outputFile)
    % Extract all vehicle IDs
    vehicleIDs = fieldnames(vehicleMetrics);

    % Prepare table headers
    dataHeaders = {'Timestamp', 'VehicleID', 'MACAddress', 'IPAddress', 'Neighbors', ...
    'NeighborNumbers', 'PowerTx', 'MCS', 'BeaconRate', 'Throughput', ...
    'Latency', 'BER', 'SER', 'PER', 'CBR', 'SNR', 'PDR', ...
    'MACSuccess', 'MACRetries', 'MACDrops', 'AvgMACDelay', ...
    'AvgMACLatency', 'AvgMACThroughput'};

    % Initialize storage for results
    results = {};

    % Iterate over timestamps
    for tIdx = 1:length(timestamps)
        currentTime = timestamps(tIdx);

        % Iterate over each vehicle
        for vIdx = 1:numel(vehicleIDs)
            vehID = enforce_vehicle_format(vehicleIDs{vIdx});
            if ~isfield(vehicleMetrics, vehID), continue; end
            metrics = vehicleMetrics.(vehID);

            % Logging
            fprintf('[LOGGING] === VEHICLE %s ===\n', vehID);
            if ~isfield(metrics, 'dataRates') || isempty(metrics.dataRates)
                fprintf('→ Throughput: NaN Mbps\n');
            end
            if ~isfield(metrics, 'neighbors') || isempty(metrics.neighbors)
                fprintf('→ Neighbors: 0 found\n');
            else
                fprintf('→ Neighbors: %d found\n', numel(metrics.neighbors));
            end
            if ~isfield(metrics, 'latencies') || isempty(metrics.latencies)
                fprintf('→ Latency avg: NaN\n');
            else
                fprintf('→ Latency avg: %.6f s\n', mean(metrics.latencies, 'omitnan'));
            end

            if ~isfield(metrics, 'totalTx') || isempty(metrics.totalTx)
                continue;
            end

            % Safe extraction with fallback
            neighbors = 'None'; neighborNumbers = 0;
            if isfield(metrics, 'neighbors') && ~isempty(metrics.neighbors)
                try
                    neighborIDs = {metrics.neighbors.id};
                    neighbors = strjoin(neighborIDs, ', ');
                    neighborNumbers = numel(neighborIDs);
                catch
                    neighbors = 'ParseErr';
                    neighborNumbers = 0;
                end
            end

            throughput = safe_avg(metrics, 'dataRates');
            latency = safe_avg(metrics, 'latencies');
            ber = safe_avg(metrics, 'ber');
            ser = safe_avg(metrics, 'ser');
            per = safe_avg(metrics, 'per');
            cbr = safe_avg(metrics, 'cbr');
            snr = safe_avg(metrics, 'snr');

            avgMacDelay = safe_avg(metrics, 'macDelay');
            avgMacLatency = safe_avg(metrics, 'macLatency');
            avgMacThroughput = safe_avg(metrics, 'macThroughput');

            if isfield(metrics, 'totalTx') && metrics.totalTx > 0
                pdr = 1 - per;
            else
                pdr = 0;
            end

            rowData = {
                currentTime,
                vehID,
                getfield_safe(metrics, 'macAddress', 'N/A'),
                getfield_safe(metrics, 'ipAddress', 'N/A'),
                neighbors,
                neighborNumbers,
                getfield_safe(metrics, 'transmissionPower', NaN),
                getfield_safe(metrics, 'MCS', NaN),
                getfield_safe(metrics, 'beaconRate', NaN),
                throughput,
                latency,
                ber,
                ser,
                per,
                cbr,
                snr,
                pdr,
                getfield_safe(metrics, 'macTxSuccess', 0),
                getfield_safe(metrics, 'macRetries', 0),
                getfield_safe(metrics, 'macDrops', 0),
                avgMacDelay,
                avgMacLatency,
                avgMacThroughput
            };

            if numel(rowData) == numel(dataHeaders)
                results(end + 1, :) = rowData; %#ok<AGROW>
            else
                fprintf('[ERROR] Skipping row due to mismatch (expected %d):\n', numel(dataHeaders));
                disp(rowData);
            end
        end
    end

    resultsTable = cell2table(results, 'VariableNames', dataHeaders);
    writetable(resultsTable, outputFile);
    disp(['[INFO] Simulation results saved to ', outputFile]);
end

function avg = safe_avg(metrics, fieldName)
    if isfield(metrics, fieldName) && ~isempty(metrics.(fieldName))
        avg = mean(metrics.(fieldName), 'omitnan');
    else
        avg = NaN;
    end
end

function val = getfield_safe(structure, fieldname, default)
    if isfield(structure, fieldname)
        val = structure.(fieldname);
    else
        val = default;
    end
end

% =========================================================================
% Number of Neighbors per Vehicles
% =========================================================================

function numNeighbors = calculate_vehicle_neighbors(fcdData, commRange, currentTime)
    % CALCULATE_VEHICLE_NEIGHBORS
    % Calculates the number of neighbors within the communication range for each vehicle.
    %
    % Inputs:
    %   fcdData - Vehicle mobility data loaded from SUMO's FCD file.
    %   commRange - Communication range of vehicles (in meters).
    %   currentTime - Current simulation time.
    %
    % Output:
    %   numNeighbors - Structure with vehicle IDs as fields and neighbor counts as values.

    % Filter vehicles at the current time
    currentFrame = fcdData([fcdData.time] == currentTime);

    % Initialize output structure
    numNeighbors = struct();

    % Loop through each vehicle
    for i = 1:length(currentFrame)
        vehID = currentFrame(i).id;
        x1 = currentFrame(i).x;
        y1 = currentFrame(i).y;

        % Initialize neighbor count
        neighborCount = 0;

        % Check distances to other vehicles
        for j = 1:length(currentFrame)
            if i == j
                continue; % Skip self-comparison
            end

            x2 = currentFrame(j).x;
            y2 = currentFrame(j).y;

            % Calculate distance
            distance = sqrt((x1 - x2)^2 + (y1 - y2)^2);

            % Increment neighbor count if within communication range
            if distance <= commRange
                neighborCount = neighborCount + 1;
            end
        end

        % Store neighbor count for the current vehicle
        numNeighbors.(vehID) = neighborCount;
    end
end


