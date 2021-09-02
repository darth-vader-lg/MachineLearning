using System;
using System.Linq;

namespace Microsoft.ML.Data
{
   /// <summary>
   /// Estensioni alle annotazioni di schema
   /// </summary>
   public static class SchemaAnnotationsExtensions
   {
      #region Methods
      /// <summary>
      /// Restituisce gli slot di una colonna
      /// </summary>
      /// <param name="column">Colonna</param>
      /// <returns>L'array di nomi di slot</returns>
      public static string[] GetSlotNames(this DataViewSchema.Column column)
      {
         var slotNames = default(VBuffer<ReadOnlyMemory<char>>);
         column.GetSlotNames(ref slotNames);
         return slotNames.GetValues().ToArray().Select(s => s.ToString()).ToArray();
      }
      #endregion
   }
}
