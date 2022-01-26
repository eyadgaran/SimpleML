'''
Persistence Locations

These are the primary classes responsible for transport of serialized files

Convention is to use a serializer to serialize python objects into the system
temp folder first and then call a location transporter to copy to the final
storage place
'''

__author__ = 'Elisha Yadgaran'
