using MachineLearning.ModelZoo;
using System;

namespace SmartDictionary
{
   class Program
   {
      static void Main(string[] args)
      {
         // Create the dictionary
         var dictionary = new SmartDictionary<string>()
         {
            { "this is a house", "house" },
            { "this is a car", "car" },
            { "this is a window", "window" },
         };

         // Test set of keys
         var similarKeys = new[]
         {
            "these are houses",
            "I see a car",
            "It seems a broken window"
         };

         // Query the dictionary
         foreach (var key in similarKeys)
            Console.WriteLine($"dictionary[\"{key}\"] => {dictionary.Similar[key]}");
      }
   }
}
