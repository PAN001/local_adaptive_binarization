all:
	g++ -I/usr/include/opencv binarizewolfjolion.cpp timing.cpp -o binarizewolfjolion `pkg-config opencv --libs` -lstdc++

clean:
	rm -f binarizewolfjolion

test:
	./binarizewolfjolion


package:	clean
	rm -f x.jpg
	tar cvfz binarizewolfjolionopencv.tgz *