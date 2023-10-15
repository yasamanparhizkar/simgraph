clear all
close all

CheckNb = 40;   % Size of the checkerboard stimulus, in number of checks
CheckSize = 16; % Number of pixels on per check of the checkerboard
MovieMag = 1;   % Magnification at which movie was presented on stimulus display. Assume 1 for now. 

MovieDX = 199;  % Both of these are X and Y stim size - 1
MovieDY = 119;

TotalSize = MovieDX*MovieDY;


    %%% INPUT YOUR RFs HERE TO REPLACE SPACE FILTERS
    %%% Should be 40 x 40 x ncells
    
    Spatial = space_filters;  
    
    %%%

    for CellNb=1:size(Spatial,3)%size(EllipseCoor,2)
         CellNb

    
    
        
    %%% This deals with pixel scaling
    for ix=1:CheckSize*size(Spatial,1)
        for iy=1:CheckSize*size(Spatial,2)
            RFval2(ix,iy) = Spatial(1+floor( (ix-1)/CheckSize ),1+floor( (iy-1)/CheckSize ),CellNb);
        end
    end
    %%%
    
    %%% This assumes the same mirroring  as was in my data, since its
    %%% from the same lab so I have no reason to assume it was done
    %%% differently. Keep in mind as it stands this is a GUESS on the
    %%% orientation of the checkerboard relative to the stimulus. 
    RFval2 = RFval2';
    RFval2 = RFval2(end:-1:1,:);
    RFval2 = RFval2( (size(RFval2,1)/2-MovieDX/2) : (size(RFval2,1)/2+MovieDX/2) , (size(RFval2,2)/2-MovieDY/2) : (size(RFval2,2)/2+MovieDY/2) );
    
    %%% These are your rescaled receptive fields
    ReframedRF(:,:,CellNb) = RFval2; 
    end


