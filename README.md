# Tetra

This is a version of tetra adapted for the IRISC Bexus project. 
As it will communicate with the rest of the software the output was changed, since it will not be read bu humans. Basically, a little bit less info is printed, and in the event of failure it will return 0 to all four values returned (Ra, Dec, Roll, FoV).
These changes are generally not very beautiful.
Also, it is being adapted to work with FIT/FITS, as that is the camera output. (Kinda WIP)

1. Create a directory (i.e. Tetra).
2. Place tetra.py in the directory.
3. Place Yale's Bright Star Catalog in the directory: <a href="http://tdc-www.harvard.edu/catalogs/BSC5" target="_blank">BSC5</a>
4. Create a subdirectory called 'pics'
5. Place images in 'pics' such as this one: <a href="http://i.imgur.com/7qPnoi1.jpg" target="_blank">Aurora</a>
6. Run 'python tetra.py'

The first run may take a while as it needs to generate the catalog.  From then on, the majority of the runtime will be taken up by loading the catalog into memory and image processing.
