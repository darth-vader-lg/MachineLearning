using Newtonsoft.Json;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;

namespace MachineLearning.Util
{
   /// <summary>
   /// Item di un file pbtxt
   /// </summary>
   internal class PbtxtItem
   {
      [SuppressMessage("Style", "IDE1006:Stili di denominazione", Justification = "<Minuscolo necessario alla deserializzazione>")]
      public string name { get; set; }
      [SuppressMessage("Style", "IDE1006:Stili di denominazione", Justification = "<Minuscolo necessario alla deserializzazione>")]
      public int id { get; set; }
      [SuppressMessage("Style", "IDE1006:Stili di denominazione", Justification = "<Minuscolo necessario alla deserializzazione>")]
      public string display_name { get; set; }
   }
   /// <summary>
   /// Elenco di items di un fie pbtxt
   /// </summary>
   internal class PbtxtItems
   {
      public List<PbtxtItem> Items { get; set; }
   }
   /// <summary>
   /// Parser del file pbtxt
   /// </summary>
   internal class PbtxtParser
   {
      internal static PbtxtItems ParsePbtxtFile(string filePath)
      {
         string line;
         var newText = "{\"items\":[";

         using (var reader = new System.IO.StreamReader(filePath)) {
            while ((line = reader.ReadLine()) != null) {
               var newline = string.Empty;
               if (line.Contains("{")) {
                  newline = line.Replace("item", "").Trim();
                  //newText += line.Insert(line.IndexOf("=") + 1, "\"") + "\",";
                  newText += newline;
               }
               else if (line.Contains("}")) {
                  newText = newText.Remove(newText.Length - 1);
                  newText += line;
                  newText += ",";
               }
               else {
                  newline = line.Replace(":", "\":").Trim();
                  newline = "\"" + newline;// newline.Insert(0, "\"");
                  newline += ",";

                  newText += newline;
               }
            }
            newText = newText.Remove(newText.Length - 1);
            newText += "]}";
            reader.Close();
         }
         var items = JsonConvert.DeserializeObject<PbtxtItems>(newText);
         return items;
      }
   }
}
